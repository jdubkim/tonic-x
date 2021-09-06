

def check_environment_type(environment, environment_type):
    environment = environment.environment

    while True:
        try:
            # Unwrap the environment
            environment = environment.env
        except AttributeError:
            break
    # Check if the environment matches the given type.
    return isinstance(environment, environment_type)
