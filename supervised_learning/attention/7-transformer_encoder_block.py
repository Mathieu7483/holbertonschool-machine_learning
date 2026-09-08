#!/usr/bin/env python3
"""Create a class EncoderBlock that inherits
from tensorflow.keras.layers.Layer to create
an encoder block for a transformer"""
import tensorflow as tf


class EncoderBlock(tf.keras.layers.Layer):
    """Class EncoderBlock that inherits from Layer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Constructor for the class EncoderBlock."""
        super(EncoderBlock, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=h,
                                                      key_dim=dm)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Method that performs the forward pass."""
        attn_output, _ = self.mha(x, x, x, attention_mask=mask,
                                  return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        hidden_output = self.dense_hidden(out1)
        output = self.dense_output(hidden_output)
        output = self.dropout2(output, training=training)
        out2 = self.layernorm2(out1 + output)

        return out2
