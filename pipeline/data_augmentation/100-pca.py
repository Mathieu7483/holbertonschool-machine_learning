#!/usr/bin/env python3
"""Performs PCA color augmentation on an image as in AlexNet."""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation on a 3D image tensor.

    Args:
        image (tf.Tensor): 3D tensor of shape (H, W, 3).
        alphas (list or tuple): 3 factors used to scale
            eigenvectors/eigenvalues.

    Returns:
        tf.Tensor: Augmented image with the same shape and dtype as image.
    """
    orig_dtype = image.dtype
    img_float = tf.image.convert_image_dtype(image, tf.float32)

    # Flatten pixels to (N, 3) and center around channel mean
    pixels = tf.reshape(img_float, [-1, 3])
    n_pixels = tf.cast(tf.shape(pixels)[0], tf.float32)
    mean = tf.reduce_mean(pixels, axis=0, keepdims=True)
    centered = pixels - mean

    # 3x3 covariance matrix and eigendecomposition
    cov = tf.matmul(centered, centered, transpose_a=True) / n_pixels
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = tf.argsort(eigenvalues, direction='DESCENDING')
    eigenvalues = tf.gather(eigenvalues, idx)
    eigenvectors = tf.gather(eigenvectors, idx, axis=1)

    # Compute perturbation: sum_i (alpha_i * eigenvalue_i * eigenvector_i)
    alphas_tensor = tf.cast(alphas, tf.float32)
    scaled_eigvals = alphas_tensor * eigenvalues
    perturbation = tf.linalg.matvec(eigenvectors, scaled_eigvals)

    # Add perturbation to image, clip to valid range,restore original dtype
    augmented = tf.reshape(img_float + perturbation, tf.shape(image))
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)
    return tf.image.convert_image_dtype(augmented, orig_dtype)
