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
    @gin.configurable(module='DictObservationEncoder')
    def initialize(self, observation_normalizer=None,
                   keywords=None):
        assert keywords is not None
        self.observation_normalizer = observation_normalizer
        self.keywords = keywords

    def call(self, observations):
        # If observation is given as a list of dictionaries,
        # convert them into a dictionary of lists
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        observations = [observations[keyword] for keyword in self.keywords]

        return tf.concat(observations, axis=-1)


@gin.configurable
class DictObservationActionEncoder(tf.keras.Model):
    @gin.configurable(module='DictObservationActionEncoder')
    def initialize(self, observation_normalizer=None,
                   keywords=None):
        assert keywords is not None
        self.observation_normalizer = observation_normalizer
        self.keywords = keywords

    def call(self, observations, actions):

        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)

        observations = [observations[keyword] for keyword in self.keywords]
        observations.append(actions)

        return tf.concat(observations, axis=-1)
