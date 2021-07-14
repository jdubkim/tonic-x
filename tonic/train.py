'''Script used to train agents.'''

from absl import app, flags
import gin

import tonic


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


@gin.configurable
def train(agent, environment, trainer, before_training, after_training,
          parallel, sequential, cfg):
    '''Trains an agent on an environment.'''

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.Environment(
        _environment, worker_groups=parallel,
        workers_per_group=sequential)
    test_environment = tonic.environments.Environment(_environment)

    # Initialize the logger to save data to the path environment/name/seed.
    tonic.logger.initialize(script_path=__file__,
                            config=cfg)

    # Build the trainer.
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment)

    # Run some code before training.
    if before_training:
        exec(compile(open(before_training).read(), before_training, 'exec'))

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(compile(open(after_training).read(), before_training, 'exec'))
        # os.system('python ' + after_training)


def main(argv):

    gin_file = FLAGS.gin_file
    gin_param = FLAGS.gin_param

    # Parse gin configurations
    gin.parse_config_files_and_bindings(gin_file, gin_param)

    # Store configs
    cfg = gin.config_str()

    # Parse configurations to train function
    with gin.unlock_config():
        gin.parse_config_file('tonic/configs/train.gin')

    train(cfg=cfg)


if __name__ == '__main__':
    app.run(main)
