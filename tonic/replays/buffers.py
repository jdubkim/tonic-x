from collections import deque
from typing import Dict

import gin
import numpy as np


@gin.configurable
class Buffer:
    '''Replay storing a large number of transitions for off-policy learning
    and using n-step returns.'''

    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=50, batch_size=100,
        discount_factor=0.99, steps_before_batches=int(1e4),
        steps_between_batches=50
    ):
        self.max_size = size
        self.num_steps = num_steps
        self.batch_iterations = batch_iterations
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.steps_before_batches = steps_before_batches
        self.steps_between_batches = steps_between_batches

    def initialize(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.buffers = None
        self.index = 0
        self.size = 0
        self.steps = 0

    def ready(self):
        if self.steps < self.steps_before_batches:
            return False
        return self.steps % self.steps_between_batches == 0

    def store(self, **kwargs):
        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        kwargs.pop('infos')

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1

    def accumulate_n_steps(self, kwargs):
        rewards = kwargs['rewards']
        next_observations = kwargs['next_observations']
        discounts = kwargs['discounts']
        masks = np.ones(self.num_workers, np.float32)

        for i in range(min(self.size, self.num_steps - 1)):
            index = (self.index - i - 1) % self.max_size
            masks *= (1 - self.buffers['resets'][index])
            new_rewards = (self.buffers['rewards'][index] +
                           self.buffers['discounts'][index] * rewards)
            self.buffers['rewards'][index] = (
                (1 - masks) * self.buffers['rewards'][index] +
                masks * new_rewards)
            new_discounts = self.buffers['discounts'][index] * discounts
            self.buffers['discounts'][index] = (
                (1 - masks) * self.buffers['discounts'][index] +
                masks * new_discounts)
            self.buffers['next_observations'][index] = (
                (1 - masks)[:, None] *
                self.buffers['next_observations'][index] +
                masks[:, None] * next_observations)

    def get(self, *keys):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            yield {k: self.buffers[k][rows, columns] for k in keys}


@gin.configurable
class DictBuffer(Buffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=50, batch_size=100, 
        discount_factor=0.99, steps_before_batches=int(1e4), 
        steps_between_batches=50
    ):
        super(DictBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor, 
                                        steps_before_batches, 
                                        steps_between_batches)

    def _unpack_dict_observations(self, kwargs):
        # Unpack observations
        observations = kwargs.pop('observations')
        for key in observations.keys():
            kwargs[key] = observations[key]

        # Store observation keys
        self.observation_keys = observations.keys()

        # Unpack next_observations
        next_observations = kwargs.pop('next_observations')
        for key in next_observations.keys():
            kwargs['next_'+key] = next_observations[key]

        return kwargs

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        kwargs.pop('infos')

        kwargs = self._unpack_dict_observations(kwargs)
        
        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])

            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size,) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1

    def get(self, *keys):
        '''Get batches from named buffers.'''
        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = self.np_random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers

            transitions = {}

            for key in keys:
                # Zip dictionary observations 
                if key == 'observations':
                    transitions[key] = {
                        k: self.buffers[k][rows, columns] \
                            for k in self.observation_keys}

                # Zip dictionary observations 
                elif key == 'next_observations':
                    transitions[key] = {
                        k: self.buffers["next_" + k][rows, columns] \
                            for k in self.observation_keys}
                else:
                    transitions[key] = self.buffers[key][rows, columns]

            yield transitions


@gin.configurable
class HerBufferOptim(DictBuffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=40,
        batch_size=2048, discount_factor=0.95,
        steps_before_batches=int(1e4), steps_between_batches=50,
        goal_selection_strategy='future', replay_k=4,
        max_timesteps=50, reward_function=None,
        handle_timeout_termination=False
    ):
        super(HerBufferOptim, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor,
                                        steps_before_batches, 
                                        steps_between_batches)
        
        self.goal_selection_strategy = goal_selection_strategy
        self.replay_k = replay_k

        # compute ratio between HER replays and regular replays
        self.her_ratio = 1 - (1.0 / (1 + self.replay_k))

        self.handle_timeout_termination = handle_timeout_termination

        self.reward_function = reward_function

    def initialize(self, seed):
        super().initialize(seed)
        self.full = False
    
    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        # Unpack dictionary observations
        kwargs = self._unpack_dict_observations(kwargs)

        infos = kwargs.pop('infos')

        # Create the named buffers
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_size, ) + np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

            # Create a list to store indices of episodes
            self.episode_reset_indices = [[] for _ in range(self.num_workers)]

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.steps += 1

        for i, reset in enumerate(kwargs['resets']):
            if reset:
                self.store_episode(i)

    def store_episode(self, env_id):
        self.episode_reset_indices[env_id].append(self.steps)

    @property
    def n_episodes_stored(self):
        return len(self.episode_reset_indices[0])

    def get(self, *keys):
        '''Get batches from named buffers.'''
        for _ in range(self.batch_iterations):
            sampled_transitions = self.sample_transitions()

            transitions = {}
            for key in keys:
                # Pack dictionary observations
                if key == 'observations':
                    transitions[key] = {k: sampled_transitions[k][:, 0]
                                        for k in self.observation_keys}
                # Pack dictionary observations
                elif key == 'next_observations':
                    transitions[key] = {k: sampled_transitions["next_"+k][:, 0]
                                        for k in self.observation_keys}
                else:
                    transitions[key] = sampled_transitions[key][:, 0]

            yield transitions

    def sample_episodes_indices(self):
        n_episodes = sum(len(episodes) 
                         for episodes in self.episode_reset_indices)
        n_episodes_per_env = np.array([len(episodes) 
                              for episodes in self.episode_reset_indices])
        n_episodes_acc = np.array([sum(n_episodes_per_env[:i+1]) 
                          for i in range(len(n_episodes_per_env))])
        episodes_env = np.array([sum([[i for _ in episode] for (i, episode) in \
            enumerate(self.episode_reset_indices)], [])])
        
        # Randomly select episodes to use.
        episode_indices = self.np_random.randint(0, n_episodes, self.batch_size)

        def index_1d_to_2d(index_1d, episode_env):
            pos = index_1d - n_episodes_acc[episode_env]

            return (episode_env, pos)

        # Calculate which environment these episodes belong to.
        episodes_indices_2d = [index_1d_to_2d(index, env_id) for index, env_id \
            in zip(episode_indices, episodes_env)]

        return episodes_indices_2d

    def sample_timesteps_indices(self, episodes_indices):

        # Get timesteps for each episode.
        episode_reset_indices_ = [[] for _ in range(len(self.episode_reset_indices))]
        for i in range(len(self.episode_reset_indices)):
            episode_reset_indices_[i].append([0] + self.episode_reset_indices[i][1:])

        episode_lengths = np.array(self.episode_reset_indices) - \
            np.array(episode_reset_indices_)

        rand = self.np_random.random(len(episodes_indices[0]))

        print("DEBUG: ")
        print("episode length: ", episode_lengths)
        print("episode indices: ", episodes_indices)
        print("rand: ", rand)
        
        return np.floor(episode_lengths[episodes_indices] * rand, dtype=np.int)

    def convert_to_buffer_indices(self, episode_indices, timestep_indices):
        
        def episode_to_timestep(episode):
            if episode == 0:
                return 0
            return self.episode_reset_indices[episode-1]

        return [(episode_to_timestep(episode) + timestep, env_id) 
         for (episode, env_id), timestep in \
             zip(episode_indices, timestep_indices)]

    def sample_transitions(self):
        # Sample episodes and timesteps
        episodes_indices = self.sample_episodes_indices()
        her_indices = \
            np.arange(self.batch_size)[: int(self.her_ratio * self.batch_size)]
        timestep_indices = self.sample_timesteps_indices(episodes_indices)
        
        buffer_indices = self.convert_to_buffer_indices(
            episodes_indices, timestep_indices)
        
        transitions = {key: self.buffers[key][buffer_indices].copy()
                       for key in self.buffers.keys()}

        new_goals = self.sample_goals(episodes_indices, her_indices, 
                                      timestep_indices)

        transitions["desired_goal"][her_indices] = new_goals

        if len(her_indices) > 0:
            transitions["rewards"][her_indices, 0] = self.reward_function(
                transitions["next_achieved_goal"][her_indices, 0],
                transitions["desired_goal"][her_indices, 0],
                None
            )

        return transitions

    def sample_goals(self, episodes_indices, her_indices, timestep_indices):
        her_episode_indices = episodes_indices[her_indices]

        episode_lengths = list(map(lambda x,y: x-y, self.episode_reset_indices,
                        [0] + self.episode_reset_indices[1:]))

        # replay with k random states from the episodes after current transitions
        if self.goal_selection_strategy == 'future':
            transitions_indices = self.np_random.randint(
                timestep_indices[her_indices] + 1,
                episode_lengths[her_episode_indices]
            )
        # replay with final state of the episodes
        elif self.goal_selection_strategy == 'final':
            transitions_indices = episode_lengths[her_episode_indices] - 1
        # replay with random state of the episodes
        elif self.goal_selection_strategy == 'episode':
            transitions_indices = self.np_random.randint(
                episode_lengths[her_episode_indices]
            )
        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy}" +
                             "for sampling goals not supported!")

        buffer_indices = self.convert_to_buffer_indices(
            her_episode_indices, transitions_indices)

        return self.buffers["achieved_goal"][buffer_indices]
        

@gin.configurable
class HerBuffer(DictBuffer):
    def __init__(
        self, size=int(1e6), num_steps=1, batch_iterations=40, batch_size=256, 
        discount_factor=0.98, steps_before_batches=int(5e4), 
        steps_between_batches=50, goal_selection_strategy='future',
        replay_k=4, max_timesteps=50, reward_function=None,
        handle_timeout_termination=False
    ):
        super(HerBuffer, self).__init__(size, num_steps, batch_iterations,
                                        batch_size, discount_factor,
                                        steps_before_batches, 
                                        steps_between_batches)
        
        self.goal_selection_strategy = goal_selection_strategy
        self.replay_k = replay_k 

        self.max_timesteps = max_timesteps
        self.max_n_episodes_stored = self.max_size // self.max_timesteps

        # compute ratio between HER replays and regular replays
        self.her_ratio = 1 - (1.0 / (1 + self.replay_k))

        self.handle_timeout_termination = handle_timeout_termination

        self.reward_function = reward_function

    def initialize(self, seed):
        super().initialize(seed)
        self.pos = 0
        self.episode_index = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_n_episodes_stored, 
                                        dtype=np.int64)

    def set_reward_function(self, reward_function, reward_normalizer=None):
        assert reward_function is not None
        self.reward_function = reward_function
        self.reward_normalizer = reward_normalizer

    def store(self, **kwargs):

        if 'terminations' in kwargs:
            continuations = np.float32(1 - kwargs['terminations'])
            kwargs['discounts'] = continuations * self.discount_factor

        kwargs = self._unpack_dict_observations(kwargs)
        infos = kwargs.pop('infos')

        # Create the named buffers.
        if self.buffers is None:
            self.num_workers = len(list(kwargs.values())[0])
            self.buffers = {}
            for key, val in kwargs.items():
                shape = (self.max_n_episodes_stored, self.max_timesteps, ) + \
                    np.array(val).shape
                self.buffers[key] = np.full(shape, np.nan, np.float32)

        # Store the new values.
        for key, val in kwargs.items():
            self.buffers[key][self.pos, self.episode_index] = val

        # Accumulate values for n-step returns.
        if self.num_steps > 1:
            self.accumulate_n_steps(kwargs)

        self.episode_index += 1
        self.steps += 1
        self.size = min(self.size + 1, self.max_size)

        done = False
        for i in range(self.num_workers):
            if kwargs['resets'][i]:
                done = True

        if done or self.episode_index >= self.max_timesteps:
            self.next_episode()
        
    def next_episode(self):

        self.episode_lengths[self.pos] = self.episode_index
        
        self.pos += 1
        if self.pos == self.max_n_episodes_stored:
            self.full = True
            self.pos = 0

        self.episode_index = 0

    @property
    def n_episodes_stored(self):
        if self.full:
            return self.max_n_episodes_stored
        return self.pos

    def get(self, *keys):
        '''Get batches from named buffers.'''

        for _ in range(self.batch_iterations):
            temp_buffers = {}
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.n_episodes_stored]
            sampled_transitions = self.sample_transitions(temp_buffers,
                                                          self.batch_size)

            transitions = {}
            for key in keys:
                if key == 'observations':
                    transitions[key] = {k: sampled_transitions[k][:, 0]
                                        for k in self.observation_keys}
                elif key == 'next_observations':
                    transitions[key] = {k: sampled_transitions[k][:, 0]
                                        for k in self.observation_keys 
                                        if k != 'desired_goal'}
                    transitions[key]['desired_goal'] = \
                        sampled_transitions['desired_goal'][:, 0]
                else:
                    transitions[key] = sampled_transitions[key][:, 0]

            yield transitions

    def sample_transitions(self, episode_batch, batch_size):

        if self.full:
            episode_indices = (
                self.np_random.randint(1, self.n_episodes_stored, batch_size) + 
                self.pos) % self.n_episodes_stored
        else:
            episode_indices = self.np_random.randint(0, self.n_episodes_stored, 
                                            batch_size)

        her_indices = np.arange(
            batch_size)[: int(self.her_ratio * batch_size)]

        ep_lengths = self.episode_lengths[episode_indices]

        # Filter episodes with 1 state: no transition available
        if self.goal_selection_strategy == 'future':
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        transitions_indices = self.np_random.randint(ep_lengths)

        transitions = {key: episode_batch[key][episode_indices, 
                                              transitions_indices].copy()
                       for key in episode_batch.keys()}

        new_goals = self.sample_goals(episode_indices, her_indices, 
                                      transitions_indices)

        transitions["desired_goal"][her_indices] = new_goals

        transitions["rewards"][her_indices, 0] = self.reward_function(
            transitions["next_achieved_goal"][her_indices, 0],
            transitions["desired_goal"][her_indices, 0],
            None)

        return transitions

    def sample_goals(self, episode_indices, her_indices, transitions_indices):
        her_episode_indices = episode_indices[her_indices]

        # replay with k random states from the episodes after current transitions
        if self.goal_selection_strategy == 'future':
            transitions_indices = self.np_random.randint(
                transitions_indices[her_indices] + 1, 
                self.episode_lengths[her_episode_indices]
            )

        # replay with final state of the episodes
        elif self.goal_selection_strategy == 'final':
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        # replay with random state of the episodes
        elif self.goal_selection_strategy == 'episode':
            transitions_indices = self.np_random.randint(
                self.episode_lengths[her_episode_indices])
        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy}" +
                             "for sampling goals not supported!")

        return self.buffers["achieved_goal"][her_episode_indices, 
                                             transitions_indices]