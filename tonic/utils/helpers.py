
def num_workers(observations):
    if isinstance(observations, dict):
        return len(list(observations.values())[0])

    return len(observations)
