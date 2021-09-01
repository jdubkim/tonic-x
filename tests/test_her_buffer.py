from collections import OrderedDict

import gin
import numpy as np
import pytest

import tonic
from tonic.replays import HerBuffer
from tonic.environments import SimpleEnv


# Constants
batch_size = 1
batch_iterations = 1


# Her Buffer
herbuffer = HerBuffer(batch_size=batch_size, batch_iterations=batch_iterations,
                      max_timesteps=10)


def test_her_buffer_initialise():

    herbuffer.initialize(seed=0)

    assert herbuffer.size == 0
    assert herbuffer.index == 0
    assert herbuffer.steps == 0
    assert herbuffer.buffers is None, \
        "buffers should not be created when initialised"


@pytest.mark.parametrize("seed", [0, 10, 20]) 
def test_her_buffer_store_one_step(seed):
    
    herbuffer.initialize(seed=seed)

    # Parse environment name using gin
    with gin.unlock_config():
        gin.bind_parameter('SimpleEnv.name', 'BitFlippingEnv')

    # Use bit flipping environment
    environment = tonic.environments.Environment(SimpleEnv, 1, 1)
    environment.initialize(seed=seed)

    observation_space = environment.observation_space
    action_space = environment.action_space
    
    # Initialise a constant agent
    agent = tonic.agents.Constant()
    agent.initialize(observation_space, action_space, seed=seed)

    # Start environment
    observations = environment.start()

    actions = agent.step(observations)
    next_observations, infos = environment.step(actions)

    assert herbuffer.buffers is None

    herbuffer.store(
        observations=observations,
        actions = actions,
        next_observations = next_observations,
        rewards=infos['rewards'],
        resets=infos['resets'],
        terminations=infos['terminations'],
    )

    # Buffer should now be created
    assert herbuffer.buffers is not None

    assert herbuffer.size == 1
    assert herbuffer.pos == 0
    assert herbuffer.episode_index == 1

    pos = herbuffer.pos
    episode_index = herbuffer.episode_index

    # Check if items are correctly stored in the buffer.
    for key in observations.keys():
        assert np.allclose(herbuffer.buffers[key][:pos+1, :episode_index], 
                           observations[key])
    for key in observations.keys():
        assert np.allclose(herbuffer.buffers['next_'+key][:pos+1, :episode_index],
                           next_observations[key])

    assert np.allclose(herbuffer.buffers['actions'][:pos+1, :episode_index], 
                       actions)

    assert herbuffer.buffers['rewards'][:pos+1, :episode_index] == \
        infos['rewards']
    assert herbuffer.buffers['terminations'][:pos+1, :episode_index] == \
        infos['terminations']

    return infos

    
@pytest.mark.parametrize("seed", [0, 10, 20])
def test_her_buffer_store_multi_steps(seed, n_steps=10):
    
    herbuffer.initialize(seed=seed)

    # Parse environment name using gin
    with gin.unlock_config():
        gin.bind_parameter('SimpleEnv.name', 'BitFlippingEnv')

    # Use bit flipping environment
    environment = tonic.environments.Environment(SimpleEnv, 1, 1)
    environment.initialize(seed=seed)

    # Give environment's reward function to her buffer.
    herbuffer.set_reward_function(environment.compute_reward())

    observation_space = environment.observation_space
    action_space = environment.action_space
    
    # Initialise a normal random agent
    agent = tonic.agents.NormalRandom()
    agent.initialize(observation_space, action_space, seed=seed)

    items = {
        'observation': [],
        'next_observation': [],
        'desired_goal': [],
        'next_desired_goal': [],
        'achieved_goal': [],
        'next_achieved_goal': [],
        'actions': [],
        'rewards': [],
        'terminations': [],
        'resets': [],
        'infos': [],
    }

    # Start environment
    observations = environment.start()
    steps = 0
    max_steps = n_steps
    n_episodes = n_steps // herbuffer.max_timesteps

    while True:
        actions = agent.step(observations)
        next_observations, infos = environment.step(actions)

        kwargs = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'rewards': infos['rewards'],
            'terminations': infos['terminations'],
            'resets': infos['resets'],
        }

        # Store an item into dictionary buffer.
        herbuffer.store(**kwargs)

        obs_keys = observations.keys()
        # Store into items dictionary for checking.
        for key in kwargs.keys():
            if key == 'observations':
                for obs_key in obs_keys:
                    items[obs_key].append(observations[obs_key])
            elif key == 'next_observations':
                for obs_key in obs_keys:
                    items['next_'+obs_key].append(next_observations[obs_key])
            else:
                items[key].append(kwargs[key])
        
        steps += 1
        
        if steps >= max_steps:
            break

        
    for key in items.keys():
        if key in ['terminations', 'resets']:
            items[key] = np.array(items[key], dtype=np.int64)
        elif key == 'infos':
            pass
        else:
            items[key] = np.array(items[key], dtype=np.float32)

    items.pop('infos')
    
    for key in items.keys():
        assert np.allclose(herbuffer.buffers[key][:n_episodes, :steps], items[key])
        
    return items

    
@pytest.mark.parametrize("seed", [0, 10, 20]) 
def test_her_buffer_compute_reward(seed):

    # Use bit flipping environment
    environment = tonic.environments.Environment(SimpleEnv, 1, 1)
    environment.initialize(seed=seed)

    reward_func = environment.compute_reward()

    # Give environment's reward function to her buffer.
    herbuffer.set_reward_function(environment.compute_reward())
    
    kwargs = test_her_buffer_store_multi_steps(seed)

    np_random = np.random.RandomState(seed)

    temp_buffers = {}
    for key in herbuffer.buffers.keys():
        temp_buffers[key] = herbuffer.buffers[key][:herbuffer.n_episodes_stored]

    


    


@pytest.mark.parametrize("seed", [0, 10, 20]) 
def test_her_buffer_get(seed):

    np_random = np.random.RandomState(seed)

    kwargs = test_her_buffer_store_multi_steps(seed)
    
    keys = ('observations', 'next_observations', 'actions', 'rewards',
            'terminations')

    # Get a reward function of an environment.
    dummy_environment = SimpleEnv()
    reward_func = dummy_environment.compute_reward

    # Get batch indices
    num_workers = herbuffer.num_workers
    max_size = herbuffer.size
    indices = [np_random.randint(max_size * num_workers, size=batch_size)
               for _ in range(batch_iterations)]
               
    i = 0

    for batch in herbuffer.get(*keys):
        rows = indices[i] // num_workers
        columns = indices[i] % num_workers

        for key in batch.keys():
            if key == 'observations':
                for obs_key in batch[key].keys():
                    item = kwargs[obs_key][rows, columns]
                    assert np.allclose(item, batch[key][obs_key])
            elif key == 'next_observations':
                for obs_key in batch[key].keys():
                    item = kwargs['next_'+obs_key][rows, columns]
                    assert np.allclose(item, 
                                       batch[key][obs_key])
            else:
                item = kwargs[key][rows, columns]
                assert np.allclose(item, batch[key])

        i += 1
