from collections import OrderedDict

import numpy as np
import pytest

from tonic.replays import DictBuffer
from common.envs import BitFlippingEnv


dictbuffer = DictBuffer(batch_size=1)


def test_dict_buffer_initialise():

    dictbuffer.initialize(seed=0)

    assert dictbuffer.size == 0
    assert dictbuffer.index == 0
    assert dictbuffer.steps == 0
    assert dictbuffer.buffers is None, "buffers should not be created when initialised"


def test_dict_buffer_store():
    
    dictbuffer.initialize(seed=0)

    # Using bit-flipping-env
    env = BitFlippingEnv(n_bits=10, continuous=True)
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, infos = env.step(action)
    action = np.array(action)

    kwargs = {
        'observations': [obs],
        'next_observations': [next_obs],
        'actions': [action],
        'rewards': np.array([reward]),
        'terminations': np.array([done]),
        'infos': [infos],
    }

    assert dictbuffer.buffers is None

    dictbuffer.store(**kwargs)

    # Buffer should now be created
    assert dictbuffer.buffers is not None

    assert dictbuffer.size == 1
    assert dictbuffer.index == 1

    dictbuffer.store(**kwargs)
    assert dictbuffer.size == 2
    assert dictbuffer.index == 2

    return kwargs


def test_dict_buffer_get():
    kwargs = test_dict_buffer_store()
    kwargs.pop('infos')
    
    keys = ('observations', 'next_observations', 'actions', 'rewards',
            'terminations')

    for batch in dictbuffer.get(*keys):
        _batch = dict(batch)
        _batch['observations'] = OrderedDict(_batch['observations'])

        print(_batch)
        print()
        print(kwargs)

        assert _batch == kwargs

test_dict_buffer_get()