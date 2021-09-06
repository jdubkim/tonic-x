import numpy as np
import pytest

import tonic
from tonic.replays import DictBuffer


# Constants
batch_size = 1
batch_iterations = 1


# Dictionary Buffer
dictbuffer = DictBuffer(batch_size=batch_size,
                        batch_iterations=batch_iterations)


def test_dict_buffer_initialise():

    dictbuffer.initialize(seed=0)

    assert dictbuffer.size == 0
    assert dictbuffer.index == 0
    assert dictbuffer.steps == 0
    assert dictbuffer.buffers is None, \
        "buffers should not be created when initialised"


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_dict_buffer_store_one_step(seed):

    dictbuffer.initialize(seed=seed)

    # Use bit flipping environment
    environment = tonic.environments.SimpleEnv('bitflipping-env', 1, 1)
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

    assert dictbuffer.buffers is None

    dictbuffer.store(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=infos['rewards'],
        resets=infos['resets'],
        terminations=infos['terminations'],
        environment_infos=infos['environment_infos']
    )

    # Buffer should now be created
    assert dictbuffer.buffers is not None

    assert dictbuffer.size == 1
    assert dictbuffer.index == 1

    # Check if items are correctly stored in the buffer.
    for key in observations.keys():
        assert np.allclose(dictbuffer.buffers[key][0], observations[key])
    for key in observations.keys():
        assert np.allclose(dictbuffer.buffers['next_'+key][0],
                           next_observations[key])

    assert np.allclose(dictbuffer.buffers['actions'][0], actions)
    assert dictbuffer.buffers['rewards'][0] == infos['rewards']
    assert dictbuffer.buffers['terminations'][0] == infos['terminations']

    return infos


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_dict_buffer_store_multi_steps(seed, n_steps=10):

    dictbuffer.initialize(seed=seed)

    # Use bit flipping environment
    environment = tonic.environments.SimpleEnv('bitflipping-env', 1, 1)
    environment.initialize(seed=seed)

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
        'environment_infos': [],
    }

    # Start environment
    observations = environment.start()
    steps = 0
    max_steps = n_steps

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
            'environment_infos': infos['environment_infos']
        }

        # Store an item into dictionary buffer.
        dictbuffer.store(**kwargs)

        obs_keys = observations.keys()
        # Store into a dictionary for later comparison.
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
        elif key == 'environment_infos':
            pass
        else:
            items[key] = np.array(items[key], dtype=np.float32)

    items.pop('environment_infos')

    for key in items.keys():
        assert np.allclose(dictbuffer.buffers[key][:steps], items[key])

    return items


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_dict_buffer_get(seed):

    dictbuffer.initialize(seed)
    np_random = np.random.RandomState(seed)

    kwargs = test_dict_buffer_store_multi_steps(seed)

    keys = ('observations', 'next_observations', 'actions', 'rewards',
            'terminations')

    # Get batch indices by using same random state as the one in buffer.
    num_workers = dictbuffer.num_workers
    max_size = dictbuffer.size
    indices = [np_random.randint(max_size * num_workers, size=batch_size)
               for _ in range(batch_iterations)]

    index = 0
    # Retrieve batches from the buffer
    for batch in dictbuffer.get(*keys):
        rows = indices[index] // num_workers
        columns = indices[index] % num_workers

        # Check if correct batches are retrieved.
        for key in batch.keys():
            if key == 'observations':
                for obs_key in batch[key].keys():
                    item = kwargs[obs_key][rows, columns]
                    assert np.allclose(item, batch[key][obs_key])
            elif key == 'next_observations':
                for obs_key in batch[key].keys():
                    item = kwargs['next_'+obs_key][rows, columns]
                    assert np.allclose(item, batch[key][obs_key])
            else:
                item = kwargs[key][rows, columns]
                assert np.allclose(item, batch[key])

        index += batch_size
