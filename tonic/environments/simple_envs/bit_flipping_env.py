from collections import OrderedDict

import numpy as np
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec


class BitFlippingEnv(GoalEnv):
    
    spec = EnvSpec("BitFlippingEnv-v0")

    def __init__(self, n_bits=10, continuous=True, max_timesteps=None):
        super(BitFlippingEnv, self).__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.MultiBinary(n_bits),
                "achieved_goal": spaces.MultiBinary(n_bits),
                "desired_goal": spaces.MultiBinary(n_bits),
            }
        )

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)

        self.continuous = continuous
        self.max_timesteps = n_bits if max_timesteps is None else max_timesteps
        self.max_episode_steps = self.max_timesteps

        self.state = None
        self.desired_goal = np.ones((n_bits,))
        

    def seed(self, seed):
        self.obs_space.seed(seed)

    def reset(self):
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def _get_obs(self):
        return OrderedDict(
        [
            ("observation", self.state.copy()),
            ("achieved_goal", self.state.copy()),
            ("desired_goal", self.desired_goal.copy()),
        ]
    )

    def step(self, action):
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
            
        obs = self._get_obs()
        reward = float(self.compute_reward(
            obs["achieved_goal"], obs["desired_goal"], None))
        done = reward == 0
        self.current_step += 1
        info = {"is_success": int(done)}

        done = done or self.current_step >= self.max_timesteps

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        # achieved_goal = np.array(achieved_goal).reshape(batch_size, -1)
        # desired_goal = np.array(desired_goal).reshape(batch_size, -1)

        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self):
        pass
