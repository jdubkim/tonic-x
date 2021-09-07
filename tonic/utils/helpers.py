from gym.spaces import Discrete


def num_workers(observations):
    if isinstance(observations, dict):
        return len(list(observations.values())[0])

    return len(observations)


def action_size(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    else:
        return action_space.shape[0]
