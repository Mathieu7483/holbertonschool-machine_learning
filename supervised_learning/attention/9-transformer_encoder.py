#!/usr/bin/env python3
"""Create a class Encoder that inherits from
tensorflow.keras.layers.Layer to create
the encoder for a transformer"""
import tensorflow as tf
import numpy as np
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Class Encoder that inherits from Layer."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Constructor method."""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = self.positional_encoding(max_seq_len,
                                                            dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def positional_encoding(self, max_seq_len, dm):
        """Function that calculates the positional encoding."""
        pos_enc = np.zeros((max_seq_len, dm))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        pos_enc = pos_enc[np.newaxis, ...]
        return tf.cast(pos_enc, dtype=tf.float32)

    def call(self, x, training, mask):
        """Function that performs the forward pass."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:, :seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
