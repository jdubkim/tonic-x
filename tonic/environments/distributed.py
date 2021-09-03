'''Builders for distributed training.'''

import copy
import multiprocessing

import gin
import numpy as np


class Sequential:
    '''A group of environments used in sequence.'''

    def __init__(self, environment, max_episode_steps, workers):
        self.environments = [copy.deepcopy(environment) for _ in range(workers)]
        self.max_episode_steps = max_episode_steps
        self.observation_space = self.environments[0].observation_space
        self.action_space = self.environments[0].action_space

    def initialize(self, seed):
        for i, environment in enumerate(self.environments):
            environment.seed(seed + i)

    def start(self):
        '''Used once to get the initial observations.'''
        observations = [env.reset() for env in self.environments]
        if isinstance(observations[0], dict):
            observations = self._preprocess_dict_obs(observations)

        self.lengths = np.zeros(len(self.environments), int)

        return observations

    def step(self, actions):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.
        environment_infos = []

        for i in range(len(self.environments)):
            ob, rew, term, info = self.environments[i].step(actions[i])

            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self.max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)
            terminations.append(term)
            environment_infos.append(info)

            if reset:
                ob = self.environments[i].reset()
                self.lengths[i] = 0

            observations.append(ob)

        if isinstance(ob, dict):
            observations = self._preprocess_dict_obs(observations)
            next_observations = self._preprocess_dict_obs(next_observations)

        infos = dict(
            observations=next_observations,
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, np.bool),
            terminations=np.array(terminations, np.bool),
            environment_infos=environment_infos)
        return observations, infos

    def render(self, mode='human', *args, **kwargs):
        outs = []
        for env in self.environments:
            out = env.render(mode=mode, *args, **kwargs)
            outs.append(out)
        if mode != 'human':
            return np.array(outs)

    def _preprocess_dict_obs(self, observations):
        
        return {key: np.array([dic[key] for dic in observations])
                for key in observations[0].keys()}
        

class Parallel:
    '''A group of sequential environments used in parallel.'''

    def __init__(
        self, environment, worker_groups, workers_per_group,
        max_episode_steps
    ):
        self.environment = environment
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self.max_episode_steps = max_episode_steps

    def initialize(self, seed):
        def proc(action_pipe, index, seed):
            '''Process holding a sequential group of environments.'''
            envs = Sequential(
                self.environment, self.max_episode_steps,
                self.workers_per_group)
            envs.initialize(seed)

            observations = envs.start()
            self.output_queue.put((index, observations))

            while True:
                actions = action_pipe.recv()
                out = envs.step(actions)
                self.output_queue.put((index, out))

        dummy_environment = copy.deepcopy(self.environment)
        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False

        self.output_queue = multiprocessing.Queue()
        self.action_pipes = []

        for i in range(self.worker_groups):
            pipe, worker_end = multiprocessing.Pipe()
            self.action_pipes.append(pipe)
            group_seed = seed + i * self.workers_per_group
            process = multiprocessing.Process(
                target=proc, args=(worker_end, i, group_seed))
            process.daemon = True
            process.start()

    def start(self):
        '''Used once to get the initial observations.'''
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            index, observations = self.output_queue.get()
            observations_list[index] = observations

        if isinstance(observations, dict):
            self.observations_list = observations_list
            self.next_observations_list = self.init_next_observations(observations_list)
        else:
            self.observations_list = np.array(observations_list)
            self.next_observations_list = np.zeros_like(self.observations_list)

        self.rewards_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.float32)
        self.resets_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.bool)
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.bool)
        self.environment_infos_list = [
            [None for _ in range(self.workers_per_group)] 
            for _ in range(self.worker_groups)]

        return self.get_observations_batch(self.observations_list)

    def init_next_observations(self, observations_list):
        next_observations_list = observations_list.copy()
        for i, observation in enumerate(observations_list):
            for key in observation.keys():
                next_observations_list[i][key] = np.zeros_like(observation[key])

        return next_observations_list

    def step(self, actions):
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send(actions)

        for _ in range(self.worker_groups):
            index, (observations, infos) = self.output_queue.get()
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos['observations']
            self.rewards_list[index] = infos['rewards']
            self.resets_list[index] = infos['resets']
            self.terminations_list[index] = infos['terminations']
            self.environment_infos_list[index] = infos['environment_infos']

        observations = self.get_observations_batch(self.observations_list)

        infos = dict(
            observations=self.get_observations_batch(self.next_observations_list),
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list),
            environment_infos=sum(self.environment_infos_list, []))
        return observations, infos

    def get_observations_batch(self, observation_list):
        if isinstance(self.observation_space, dict) or \
            isinstance(self.observation_space.sample(), dict):
            return self._preprocess_dict_obs(observation_list)
        else:
            return np.concatenate(observation_list)

    def _preprocess_dict_obs(self, observations):
        ''' Convert list of dictionary observations to dictionary of lists.''' 
        dict_obs = {k: [] for k in observations[0].keys()}

        for key in dict_obs.keys():
            for dic in observations:
                for obs in dic[key]:
                    dict_obs[key].append(obs)
            dict_obs[key] = np.array(dict_obs[key])

        return dict_obs


@gin.configurable
def Environment(builder, worker_groups=1, workers_per_group=1): # noqa
    '''Distributes workers over parallel and sequential groups.'''
    dummy_environment = builder()
    max_episode_steps = dummy_environment.max_episode_steps
    del dummy_environment

    if worker_groups < 2:
        return Sequential(
            builder, max_episode_steps=max_episode_steps,
            workers=workers_per_group)

    return Parallel(
        builder, worker_groups=worker_groups,
        workers_per_group=workers_per_group,
        max_episode_steps=max_episode_steps)
