

def check_environment_type(environment, environment_type):
    environment = environment.environment

    # Check if the environment matches the given type.
    return isinstance(environment.unwrapped, environment_type)
