import copy 

import gin
import tensorflow as tf


@gin.configurable
class ObservationEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return observations


@gin.configurable
class ObservationActionEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None):
        self.observation_normalizer = observation_normalizer

    def call(self, observations, actions):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        return tf.concat([observations, actions], axis=-1)


@gin.configurable
class DictObservationEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None, 
                   keywords=['observation', 'desired_goal']):

        self.observation_normalizer = observation_normalizer
        self.keywords = keywords
        
    def call(self, observations):
        # If observation is given as a batch, conver them into a dictionary of list
        if isinstance(observations, list):
            observations = {k: [dic[k] for dic in observations] for k in observations[0]}
        
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        observations = [observations[keyword] for keyword in self.keywords]
        observations = tf.concat(observations, axis=-1)

        return observations


@gin.configurable
class DictObservationActionEncoder(tf.keras.Model):
    def initialize(self, observation_normalizer=None,
                   keywords=['observation', 'desired_goal']):
        self.observation_normalizer = observation_normalizer
        self.keywords = keywords

    def call(self, observations, actions):
        # If observation is given as a batch, conver them into a dictionary of list
        if isinstance(observations, list):
            observations = {k: [dic[k] for dic in observations] for k in observations[0]}

        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        observations = [observations[keyword] for keyword in self.keywords]
        observations = tf.concat(observations, axis=-1)

        return tf.concat([observations, actions], axis=-1)