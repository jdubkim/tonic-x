import numpy as np
import pytest

import tonic
from tonic.replays import HerBuffer


# Constants
batch_size = 1
batch_iterations = 1


# Her Buffer
herbuffer = HerBuffer(batch_size=batch_size, batch_iterations=batch_iterations)


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_her_buffer_initialise(seed):

    herbuffer.initialize(seed)

    assert herbuffer.size == 0
    assert herbuffer.index == 0
    assert herbuffer.steps == 0
    assert herbuffer.buffers is None, \
        "buffers should not be created when initialised"


@pytest.mark.parametrize("seed", [0, 10, 20])
def test_her_buffer_store_one_step(seed):

    herbuffer.initialize(seed=seed)

    # Use bit flipping environment
    environment = tonic.environments.SimpleEnv('bitflipping-env', n_bits=10)
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
        actions=actions,
        next_observations=next_observations,
        rewards=infos['rewards'],
        resets=infos['resets'],
        terminations=infos['terminations'],
        environment_infos=infos['env_infos']
    )

    # Buffer should now be created
    assert herbuffer.buffers is not None

    assert herbuffer.size == 1
    assert herbuffer.steps == 1
    assert herbuffer.index == 1

    index = herbuffer.index

    # Check if items are correctly stored in the buffer.
    for key in observations:
        assert np.allclose(herbuffer.buffers[key][:index],
                           observations[key])
    for key in observations:
        assert np.allclose(herbuffer.buffers['next_'+key][:index],
                           next_observations[key])

    assert np.allclose(herbuffer.buffers['actions'][:index],
                       actions)

    assert herbuffer.buffers['rewards'][:index] == \
        infos['rewards']
    assert herbuffer.buffers['terminations'][:index] == \
        infos['terminations']

    return infos


@pytest.mark.parametrize("seed", [0, 10, 20])
@pytest.mark.parametrize("max_steps", [10, 20, 30])
def test_her_buffer_store_multi_steps(seed, max_steps):

    herbuffer.initialize(seed=seed)

    # Use bit flipping environment
    environment = tonic.environments.SimpleEnv('bitflipping-env', n_bits=10)
    environment.initialize(seed=seed)

    # Give environment's reward function to her buffer.
    herbuffer.set_reward_function(environment.compute_reward)

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

    while steps < max_steps:
        actions = agent.step(observations)
        next_observations, infos = environment.step(actions)

        kwargs = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'rewards': infos['rewards'],
            'terminations': infos['terminations'],
            'resets': infos['resets'],
            'environment_infos': infos['env_infos']
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

    for key in items.keys():
        if key in ['terminations', 'resets']:
            items[key] = np.array(items[key], dtype=np.int64)
        else:
            items[key] = np.array(items[key], dtype=np.float32)

    for key in items.keys():
        assert np.allclose(herbuffer.buffers[key][:steps], items[key])

    return items


@pytest.mark.parametrize("seed", [0, 10, 20])
@pytest.mark.parametrize("max_steps", [10, 20, 30])
def test_her_buffer_get(seed, max_steps):

    np_random = np.random.RandomState(seed)

    kwargs = test_her_buffer_store_multi_steps(seed, max_steps)

    keys = ('observations', 'next_observations', 'actions', 'rewards',
            'terminations')

    # Get batch indices
    num_workers = herbuffer.num_workers
    max_size = herbuffer.size
    indices = [np_random.randint(max_size * num_workers, size=batch_size)
               for _ in range(batch_iterations)]

    index = 0
    for batch in herbuffer.get(*keys):
        rows = indices[index] // num_workers
        columns = indices[index] % num_workers

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
