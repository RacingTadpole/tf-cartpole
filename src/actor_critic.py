import tensorflow as tf
from tensorflow import keras

from typing import Tuple


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = keras.layers.Dense(num_hidden_units, activation="relu")
        self.actor = keras.layers.Dense(num_actions)
        self.critic = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
