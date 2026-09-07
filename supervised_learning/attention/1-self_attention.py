#!/usr/bin/env python3
"""Create a class SelfAttention that inherits
from tensorflow.keras.layers.Layer to calculate
the attention for machine translation based on
this paper: https://arxiv.org/pdf/1409.0473.pdf"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Class SelfAttention that inherits from tensorflow.keras.layers.Layer."""

    def __init__(self, units):
        """Constructor method."""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Method that performs the forward propagation."""
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
