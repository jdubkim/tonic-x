'''Script used to play with trained agents.'''

import argparse
import os

from absl import app, flags
import gin
import numpy as np

import tonic  # noqa
from tonic.environments import Bullet, ControlSuite, Gym


flags.DEFINE_multi_string(
    'gin_file', [], 'List of paths to gin configuration files.'
                    ' Example: "tonic/configs/train.gin".'
)
flags.DEFINE_multi_string(
    'gin_param', [],
    'Gin parameter bindings to override the values in the configuration '
    'files. Example: "AGENT = @A2C()", "environment_name = "AntBulletEnv-v0".'
)

FLAGS = flags.FLAGS


def load_default_agent():
    return tonic.agents.NormalRandom()


def load_default_environment():
    return Bullet('BulletAntEnv-v0')


def play_gym(agent, environment):
    '''Launches an agent in a Gym-based environment.'''

    observations = environment.start()
    environment.render()

    score = 0
    length = 0
    min_reward = float('inf')
    max_reward = -float('inf')
    episodes = 0

    while True:
        actions = agent.test_step(observations)
        observations, infos = environment.step(actions)
        agent.test_update(**infos)
        environment.render()

        reward = infos['rewards'][0]
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        length += 1

        if infos['resets'][0]:
            episodes += 1

            print()
            print('Episodes:', episodes)
            print('Score:', score)
            print('Length:', length)
            print('Min reward:', min_reward)
            print('Max reward:', max_reward)

            score = 0
            length = 0


def play_control_suite(agent, environment):
    '''Launches an agent in a DeepMind Control Suite-based environment.'''

    from dm_control import viewer

    class Wrapper:
        '''Wrapper used to plug a Tonic environment in a dm_control viewer.'''

        def __init__(self, environment):
            self.environment = environment
            self.unwrapped = environment.unwrapped
            self.action_spec = self.unwrapped.environment.action_spec
            self.physics = self.unwrapped.environment.physics
            self.infos = None
            self.episodes = 0

        def reset(self):
            '''Mimics a dm_control reset for the viewer.'''

            self.observations = self.environment.reset()[None]

            self.score = 0
            self.length = 0

            return self.unwrapped.last_time_step

        def step(self, actions):
            '''Mimics a dm_control step for the viewer.'''

            ob, rew, term, _ = self.environment.step(actions)
            self.score += rew
            self.length += 1
            timeout = self.length == self.environment.max_episode_steps
            done = term or timeout

            if done:
                print()
                self.episodes += 1
                print('Episodes:', self.episodes)
                print('Score:', self.score)
                print('Length:', self.length)

            self.observations = ob[None]
            self.infos = dict(
                observations=ob[None], rewards=np.array([rew]),
                resets=np.array([done]), terminations=[term])

            return self.unwrapped.last_time_step

    # Wrap the environment for the viewer.
    environment = Wrapper(environment)

    def policy(timestep):
        '''Mimics a dm_control policy for the viewer.'''

        if environment.infos is not None:
            agent.test_update(**environment.infos)
        return agent.test_step(environment.observations)

    # Launch the viewer with the wrapped environment and policy.
    viewer.launch(environment, policy)


@gin.configurable
def play(path='.', checkpoint='last', seed=10, agent=None, environment=None):
    '''Reloads an agent and an environment from a previous experiment.'''

    tonic.logger.log(f'Loading experiment from {path}')

    # If agent and environment not specified, load default agent and
    # environment
    if not agent:
        agent = load_default_agent()
    if not environment:
        environment = load_default_environment()

    # Use no checkpoint, the agent is freshly created.
    if checkpoint == 'none':
        checkpoint_path = None
        tonic.logger.log('Not loading any weights')

    else:
        print("DEBUG: ", path)
        checkpoint_path = os.path.join(path, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            tonic.logger.error(f'{checkpoint_path} is not a directory')
            checkpoint_path = None

        # List all the checkpoints.
        checkpoint_ids = []
        for file in os.listdir(checkpoint_path):
            if file[:5] == 'step_':
                checkpoint_id = file.split('.')[0]
                checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
            # Use the last checkpoint.
            if checkpoint == 'last':
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(
                    checkpoint_path, f'step_{checkpoint_id}')

            # Use the specified checkpoint.
            else:
                checkpoint_id = int(checkpoint)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(
                        checkpoint_path, f'step_{checkpoint_id}')
                else:
                    tonic.logger.error(f'Checkpoint {checkpoint_id} '
                                       f'not found in {checkpoint_path}')
                    checkpoint_path = None

        else:
            tonic.logger.error(f'No checkpoint found in {checkpoint_path}')
            checkpoint_path = None

    # Build the environment
    _environment = environment
    environment = tonic.environments.Environment(_environment)
    environment.initialize(seed)

    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space)

    # Load the weights of the agent form a checkpoint.
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Play with the agent in the environment.
    if _environment.__name__ == 'ControlSuite':
        play_control_suite(agent, environment)
    else:
        if _environment.__name__ == 'Bullet':
            environment.render()
        play_gym(agent, environment)


def main(argv):

    gin_file = FLAGS.gin_file
    gin_param = FLAGS.gin_param

    # Parse gin configurations
    gin.parse_config_files_and_bindings(gin_file, gin_param)

    # Receive path from gin file
    config = [file for file in gin_file if 'config.gin' in file][0]
    path = os.path.dirname(config)

    # Parse configurations to play function
    with gin.unlock_config():
        gin.parse_config_file('tonic/configs/play.gin')

    play(path)


if __name__ == '__main__':
    app.run(main)
