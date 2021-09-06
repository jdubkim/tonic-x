import copy

import abc
import gin
import gym

from tonic import environments


class Environment(abc.ABC):
    def __init__(self, name, worker_groups=1, workers_per_group=1, *args,
                 **kwargs):

        self.name = name
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group

        self.dummy_environment = self.create_environment(name, *args, **kwargs)
        self.environment = self.environment_wrapper(self.dummy_environment)

        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

        super(Environment, self).__init__()

    @abc.abstractmethod
    def create_environment(self, name, *args, **kwargs):
        pass

    def initialize(self, seed):
        self.distribute_environment()
        self.distributed_environment.initialize(seed)

    def start(self):
        return self.distributed_environment.start()

    def step(self, actions):
        return self.distributed_environment.step(actions)

    def render(self, mode, *args, **kwargs):
        return self.distributed_environment.render(mode, *args, **kwargs)

    @gin.configurable
    def environment_wrapper(
        self, environment, terminal_timeouts=False, time_feature=False,
        max_episode_steps='default', scaled_actions=True
    ):
        '''Wrap an environment.
        Time limits can be properly handled with terminal_timeouts=False or
        time_feature=True, see https://arxiv.org/pdf/1712.00378.pdf for more
        details.
        '''

        # Get the default time limit.
        if max_episode_steps == 'default':
            max_episode_steps = environment._max_episode_steps

        # Remove the TimeLimit wrapper if needed.
        if not terminal_timeouts:
            assert type(environment) == gym.wrappers.TimeLimit, environment
            environment = environment.env

        # Add time as a feature if needed.
        if time_feature:
            environment = environments.wrappers.TimeFeature(
                environment, max_episode_steps)

        # Scale actions from [-1, 1]^n to the true action space if needed.
        if scaled_actions:
            environment = environments.wrappers.ActionRescaler(environment)

        environment.max_episode_steps = max_episode_steps

        return environment

    @gin.configurable
    def distribute_environment(self):
        '''Distributes workers over parallel and sequential groups.'''
        dummy_environment = copy.deepcopy(self.environment)
        max_episode_steps = dummy_environment.max_episode_steps
        del dummy_environment

        if self.worker_groups < 2:
            self.distributed_environment = environments.Sequential(
                self.environment, max_episode_steps=max_episode_steps,
                workers=self.workers_per_group)

        self.distributed_environment = environments.Parallel(
            self.environment, worker_groups=self.worker_groups,
            workers_per_group=self.workers_per_group,
            max_episode_steps=max_episode_steps)


@gin.configurable
class Gym(Environment):
    def __init__(self, name, worker_groups=1, workers_per_group=1, *args,
                 **kwargs):
        super(Gym, self).__init__(name, worker_groups, workers_per_group,
                                  *args, **kwargs)

    def create_environment(self, name, *args, **kwargs):
        return gym.make(name, *args, **kwargs)

    @property
    def compute_reward(self):
        """ Returns a reward function of an environment. """
        return self.environment.compute_reward


@gin.configurable
class Bullet(Environment):
    def __init__(self, name, worker_groups=1, workers_per_group=1, *args,
                 **kwargs):
        super(Bullet, self).__init__(name, worker_groups, workers_per_group,
                                     *args, **kwargs)

    def create_environment(self, name, *args, **kwargs):
        import pybullet_envs  # noqa
        return gym.make(name, *args, **kwargs)


@gin.configurable
class ControlSuite(Environment):
    def __init__(self, name, worker_groups=1, workers_per_group=1, *args,
                 **kwargs):
        super(Bullet, self).__init__(name, worker_groups, workers_per_group,
                                     *args, **kwargs)

    def create_environment(self, name, *args, **kwargs):
        domain, task = name.split('-')
        environment = environments.ControlSuiteEnvironment(
            domain_name=domain, task_name=task, *args, **kwargs)
        return gym.wrappers.TimeLimit(environment, 1000)


@gin.configurable
class SimpleEnv(Environment):
    def __init__(self, name, worker_groups=1, workers_per_group=1, *args,
                 **kwargs):
        super(SimpleEnv, self).__init__(name, worker_groups, workers_per_group,
                                        *args, **kwargs)

    def create_environment(self, name, *args, **kwargs):
        environment = environments.make_simple_env(name, *args, **kwargs)
        return gym.wrappers.TimeLimit(environment,
                                      environment._max_episode_steps)

    @property
    def compute_reward(self):
        """ Returns a reward function of an environment. """
        return self.environment.compute_reward
