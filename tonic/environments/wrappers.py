'''Environment wrappers.'''
from collections import deque

import gym
import numpy as np


class ActionRescaler(gym.ActionWrapper):
    '''Rescales actions from [-1, 1]^n to the true action space.
    The baseline agents return actions in [-1, 1]^n.'''

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)
        super().__init__(env)
        high = np.ones(env.action_space.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-high, high=high)
        true_low = env.action_space.low
        true_high = env.action_space.high
        self.bias = (true_high + true_low) / 2
        self.scale = (true_high - true_low) / 2

    def action(self, action):
        return self.bias + self.scale * np.clip(action, -1, 1)


class TimeFeature(gym.Wrapper):
    '''Adds a notion of time in the observations.
    It can be used in terminal timeout settings to get Markovian MDPs.
    '''

    def __init__(self, env, max_steps, low=-1, high=1):
        super().__init__(env)
        dtype = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, low).astype(dtype),
            high=np.append(self.observation_space.high, high).astype(dtype))
        self.max_episode_steps = max_steps
        self.steps = 0
        self.low = low
        self.high = high

    def reset(self, **kwargs):
        self.steps = 0
        observation = self.env.reset(**kwargs)
        observation = np.append(observation, self.low)
        return observation

    def step(self, action):
        assert self.steps < self.max_episode_steps
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        prop = self.steps / self.max_episode_steps
        v = self.low + (self.high - self.low) * prop
        observation = np.append(observation, v)
        return observation, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    '''Make end-of-life = end-of-episode, but only reset on true game over.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        if self.was_real_done:
            observation = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal state
            observation, _, _, _ = self.env.step(0)

        self.lives = self.env.unwrapped.ale.lives()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.was_real_done = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True

        self.lives = lives
        return observation, reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class FireResetEnv(gym.Wrapper):
    '''Take action on reset for environments that are fixed until firing
    '''
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        observation, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        observation, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)

        return observation

    def step(self, action):
        return self.env.step(action)


class FrameStack(gym.Wrapper):
    '''Stack frames of observations.
    Returns lazy array, which is more memory efficient.
    '''
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        observation_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(
                observation_shape[:-1] + (observation_shape[-1] * n_frames,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(observation)
        return self._get_observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class LazyFrames:
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[-1]

    def frame(self, i):
        return self._force()[..., i]
