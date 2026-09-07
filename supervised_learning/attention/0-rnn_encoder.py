#!/usr/bin/env python3
"""Create a class RNNEncoder that inherits from
tensorflow.keras.layers.Layer to encode for
machine translation:"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Class RNNEncoder that inherits from tensorflow.keras.layers.Layer."""

    def __init__(self, vocab, embedding, units, batch):
        """Constructor method."""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """Method that initializes the hidden state."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Method that performs the forward propagation."""
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=initial)
        return output, state
