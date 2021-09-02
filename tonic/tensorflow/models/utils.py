import gin
import tensorflow as tf


def default_dense_kwargs():
    return dict(
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode='fan_in', distribution='uniform'),
        bias_initializer=tf.keras.initializers.VarianceScaling(
            scale=1 / 3, mode='fan_in', distribution='uniform'))


@gin.configurable
def mlp(units, activation, dense_kwargs=None):
    if dense_kwargs is None:
        dense_kwargs = default_dense_kwargs()
    layers = [tf.keras.layers.Dense(u, activation, **dense_kwargs)
              for u in units]
    return tf.keras.Sequential(layers)


def get_observation_space(observation_space):

    if isinstance(observation_space, dict) or \
        isinstance(observation_space.sample(), dict):
        obs_shape = {k: v.shape for k, v in observation_space.spaces.items()}
    else:
        obs_shape = observation_space.shape

    return obs_shape


def get_dummy_observations(observation_shape):
    if isinstance(observation_shape, dict):
        dummy_observations = {k: tf.zeros((1,) + v) \
            for k, v in observation_shape.items()}
    else:
        dummy_observations = tf.zeros((1,) + observation_shape)

    return dummy_observations

MLP = mlp
