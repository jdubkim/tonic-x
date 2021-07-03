'''Script used to train agents.'''

import argparse
import os

from absl import app, flags
import gin

import tonic
from tonic.configs import Config


flags.DEFINE_multi_string(
        'gin_file', [], 'List of paths to gin configuration files.'
                        ' Example: "tonic/configs/agent.gin".'
    )
flags.DEFINE_multi_string(
        'gin_param', [],
        'Gin parameter bindings to override the values in the configuration '
        'files. Example: "train.seed = 10", "train.sequential = 1".'
)

FLAGS = flags.FLAGS


def train(config):
    '''Trains an agent on an environment.'''

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Build the agent.
    agent = config.agent

    # Build the train and test environments.
    _environment = config.environment
    environment = tonic.environments.distribute(
        lambda: eval(_environment), config.parallel, config.sequential)
    test_environment = tonic.environments.distribute(
        lambda: eval(_environment))

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not config.name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if config.parallel != 1 or config.sequential != 1:
            name += f'-{config.parallel}x{config.sequential}'

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join(environment_name, name, str(config.seed))
    tonic.logger.initialize(path, script_path=__file__,
                            config=gin.config_str())

    # Build the trainer.
    trainer = config.trainer
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, seed=config.seed)

    # Run some code before training.
    if config.before_training:
        exec(config.before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if config.after_training:
        exec(config.after_training)


def main(argv):
    gin_file = FLAGS.gin_file
    gin_param = FLAGS.gin_param

    gin.parse_config_files_and_bindings(gin_file, gin_param)
    # configs = gin.config_str()
    config = Config()
    train(config)


if __name__ == '__main__':
    app.run(main)
