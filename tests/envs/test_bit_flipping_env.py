from gym import spaces
import numpy as np
import pytest

from tonic.environments.simple_envs import BitFlippingEnv


def test_env_init():
    env = BitFlippingEnv(n_bits=10)
    obs = env.reset()

    assert obs['observation'].shape == (10,)
    assert obs['achieved_goal'].shape == (10,)
    assert obs['desired_goal'].shape == (10,)
    

def test_env_discrete_action():
    env = BitFlippingEnv(n_bits=10, continuous=False)
    obs = env.reset()

    assert isinstance(env.action_space, spaces.Discrete)

    action = env.action_space.sample()

    assert isinstance(action, int)

    next_obs, r, done, _ = env.step(action)

    assert not np.allclose(obs['observation'], next_obs['observation'])


def test_env_continuous_action():
    env = BitFlippingEnv(n_bits=10, continuous=True)
    obs = env.reset()

    assert isinstance(env.action_space, spaces.Box)

    action = env.action_space.sample()

    assert action.shape == obs['observation'].shape

    next_obs, r, done, _ = env.step(action)

    assert not np.allclose(obs['observation'], next_obs['observation'])


@pytest.mark.parametrize("n_bits", [10, 20, 30]) 
def test_env_timeout(n_bits):
    """ Env should timeout when steps equals number of bits. """ 
    env = BitFlippingEnv(n_bits=n_bits, continuous=False)
    env.reset()

    action = 0
    curr_steps = 0
    done = False

    while not done:
        obs, r, done, info = env.step(action)
        curr_steps += 1

    assert curr_steps == n_bits
        

def test_return_is_success_correctly():
    # Test if environment returns is_success correctly when reached desird goal
    n_bits = 10
    env = BitFlippingEnv(n_bits=n_bits, continuous=False)
    env.desired_goal = np.ones((n_bits,))

    env.reset()
    env.state = np.concatenate([np.zeros((1,)), np.ones((n_bits-1,))])

    action = 0

    obs, r, done, info = env.step(action)

    assert info['is_success'] == 1
    assert r == 0
    assert done
