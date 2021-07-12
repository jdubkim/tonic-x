import gin


@gin.configurable
class Config(object):
    """ Configuration for training and playing an agent """
    def __init__(self, agent, environment, trainer, before_training,
                 after_training, parallel, sequential, seed, name):
        self.agent = agent
        self.environment = environment
        self.trainer = trainer
        self.seed = seed
        self.name = name
