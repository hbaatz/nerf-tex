"""Collection of network layers used in the models."""

from typing import Any
import tensorflow as tf
from tensorflow.keras.layers import Layer, LeakyReLU
from math import pi

class FourierFeatures(Layer):
    """Create fourier features from input."""

    def __init__(self, n_freq_bands: int) -> None:
        super().__init__()
        self.feature_maps = []
        self.feature_maps.append(lambda x: x)

        freq_bands = 2 ** tf.range(n_freq_bands, dtype=tf.dtypes.float32)

        for freq in freq_bands:
            self.feature_maps.append(lambda x, freq=freq: tf.math.sin(freq * x))
            self.feature_maps.append(lambda x, freq=freq: tf.math.cos(freq * x))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.concat([feature_map(inputs) for feature_map in self.feature_maps], -1)

class IntegratedPositionalEncoding(Layer):
    """Integrated Positional Encoding."""

    def __init__(self, n_freq_bands: int) -> None:
        super().__init__()

        self.n_freq_bands = n_freq_bands
        self.freq_bands = 2 ** tf.range(n_freq_bands, dtype=tf.dtypes.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        y = tf.reshape(inputs[..., None, :3] * self.freq_bands[:, None], (-1, 3 * self.n_freq_bands))
        y_var = tf.reshape(inputs[..., None, 3:] * self.freq_bands[:, None]**2, (-1, 3 * self.n_freq_bands))

        return self.expected_sin(tf.concat([y, y + .5 * pi], axis=-1), tf.concat([y_var, y_var], axis=-1))

    def expected_sin(self, x, x_var):
        return tf.sin(x) * tf.exp(-.5 * x_var)