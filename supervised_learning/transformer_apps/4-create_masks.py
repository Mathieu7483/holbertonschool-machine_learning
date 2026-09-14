#!/usr/bin/env python3
"""Creates all masks for training/validation of a transformer model."""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation.
    Args:
        inputs [tf.Tensor]: contains the input sentence.
        target [tf.Tensor]: contains the target sentence.
    Returns: enc_padding_mask, combined_mask, dec_padding_mask
        enc_padding_mask [tf.Tensor]: padding mask to be applied in the encoder
        combined_mask [tf.Tensor]: combined mask to be applied in the 1st
            attention block in the decoder.
        dec_padding_mask [tf.Tensor]: padding mask to be applied in the 2nd
            attention block in the decoder.
    """
    # Encoder padding mask
    enc_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask
    dec_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target.shape[1], target.shape[1])), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = (
        dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    )
    combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
