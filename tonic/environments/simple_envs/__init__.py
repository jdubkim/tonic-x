from .bit_flipping_env import BitFlippingEnv

__all__ = [BitFlippingEnv]


# Dictionary mapping between environemnt id and environment class.
# Add a new entry here after creating a new environment.
environment_dict = {
    'BitFlippingEnv': BitFlippingEnv,
}


def make_env(name, *args, **kwargs):
    try:
        environment_builder = environment_dict[name]
    except KeyError:
        print("Environment Not found.")

    return environment_builder(*args, **kwargs)