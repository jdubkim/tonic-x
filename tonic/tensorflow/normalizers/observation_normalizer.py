import gin
import numpy as np
import tensorflow as tf


@gin.configurable
class ObservationNormalizer(tf.keras.Model):
    def __init__(self, normalizer_class):
        super(ObservationNormalizer, self).__init__()
        self.normalizer_class = normalizer_class
        
    def initialize(self, obs_shape):
        assert isinstance(obs_shape, tuple)

        self.obs_shape = obs_shape
        self.observation_normalizer = self.normalizer_class()
        self.observation_normalizer.initialize(obs_shape)

    def __call__(self, values):
        return self.observation_normalizer(values)

    def unnormalize(self, val):
        self.observation_normalizer.unnormalize(val)

    def record(self, values):
        self.observation_normalizer.record(values)

    def update(self):
        self.observation_normalizer.update()


@gin.configurable
class DictObservationNormalizer(tf.keras.Model):
    """ Stores a dict of observation normalizers for dictioanry observations."""
    def __init__(self, normalizer_class):
        super(DictObservationNormalizer, self).__init__()
        self.normalizer_class = normalizer_class
    
    def initialize(self, obs_shape):
        assert isinstance(obs_shape, dict)
        self.obs_shape = obs_shape

        self.observation_normalizer = {k: self.normalizer_class() \
            for k in self.obs_shape.keys()}
        
        for k, shape in self.obs_shape.items():
            self.observation_normalizer[k].initialize(shape)

    def __call__(self, values):
        return {k: normalizer(values[k]) \
            for k, normalizer in self.observation_normalizer.items()}

    def unnormalize(self, vals):
        return {k: normalizer.unnormalize(vals[k]) \
            for k, normalizer in self.observation_normalizer.items()}

    def record(self, values):
        for k, val in values.items():
            self.observation_normalizer[k].record(val)

    def update(self):
        for normalizer in self.observation_normalizer.values():
            normalizer.update()
