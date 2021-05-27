'''Script used to train agents.'''

import argparse
import os

from absl import app, flags
import gin

import tonic


flags.DEFINE_multi_string(
        'gin_file', [], 'List of paths to gin configuration files.'
                        ' Example: "tonic/configs/agent.gin".'
    )
flags.DEFINE_multi_string(
        'gin_binding', [],
        'Gin parameter bindings to override the values in the configuration '
        'files.'
        ' Example: "train.seed = 10", "train.sequential = 1".'
)

FLAGS = flags.FLAGS


@gin.configurable
def train(
    agent, environment,  trainer, before_training,
        after_training,
    parallel, sequential, seed, name, configs
):
    '''Trains an agent on an environment.'''

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Build the agent.
    agent = agent

    # Build the train and test environments.
    environment = tonic.environments.distribute(
        lambda: environment, parallel, sequential)
    test_environment = tonic.environments.distribute(
        lambda: environment)

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f'-{parallel}x{sequential}'

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join(environment_name, name, str(seed))
    print(configs)
    tonic.logger.initialize(path, script_path=__file__, config=configs)

    # Build the trainer.
    trainer = trainer
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, seed=seed)

    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)


def main(argv):
    print(argv)
    gin_file = FLAGS.gin_file
    gin_binding = FLAGS.gin_binding

    gin.parse_config_files_and_bindings(gin_file, gin_binding,
                                        skip_unknown=False)
    configs = gin.config_str()

    train(configs=configs)


if __name__ == '__main__':
    app.run(main)
