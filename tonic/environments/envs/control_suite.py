from collections import OrderedDict

import gin
import gym
import numpy as np


@gin.configurable
class ControlSuiteEnvironment(gym.core.Env):
    '''Turns a Control Suite environment into a Gym environment.'''

    def __init__(
        self, domain_name, task_name, task_kwargs=None, visualize_reward=True,
        environment_kwargs=None, flatten=False
    ):
        from dm_control import suite
        self.environment = suite.load(
            domain_name=domain_name, task_name=task_name,
            task_kwargs=task_kwargs, visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs)

        # Create the observation space.
        self.flatten = flatten
        observation_spec = self.environment.observation_spec()
        if flatten:
            dim = sum([np.int(np.prod(spec.shape))
                    for spec in observation_spec.values()])
            high = np.full(dim, np.inf, np.float32)
            self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        else:
            self.observation_space = observation_spec
            self.observation_space.spaces = OrderedDict(
                self.observation_space.items())

        # Create the action space.
        action_spec = self.environment.action_spec()
        self.action_space = gym.spaces.Box(
            action_spec.minimum, action_spec.maximum, dtype=np.float32)

    def seed(self, seed):
        self.environment.task._random = np.random.RandomState(seed)

    def step(self, action):
        time_step = self.environment.step(action)
        observation = time_step.observation

        if self.flatten:
            observation = _flatten_observation(observation)

        reward = time_step.reward

        # Remove terminations from timeouts.
        done = time_step.last()
        if done:
            done = self.environment.task.get_termination(
                self.environment.physics)
            done = done is not None

        self.last_time_step = time_step
        return observation, reward, done, {}

    def reset(self):
        time_step = self.environment.reset()
        self.last_time_step = time_step
        if self.flatten:
            return _flatten_observation(time_step.observation)
        return time_step.observation

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        '''Returns RGB frames from a camera.'''
        assert mode == 'rgb_array'
        return self.environment.physics.render(
            height=height, width=width, camera_id=camera_id)

            
def _flatten_observation(observation):
    '''Turns OrderedDict observations into vectors.'''
    observation = [np.array([o]) if np.isscalar(o) else o.ravel()
                   for o in observation.values()]
    return np.concatenate(observation, axis=0)

