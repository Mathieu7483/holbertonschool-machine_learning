#!/usr/bin/env python3
"""Write a function that builds the inception network as
described in Going Deeper with Convolutions (2014)."""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in Going Deeper with
    Convolutions (2014):
    The model should be compiled using Adam optimization and categorical
    cross-entropy loss.
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    conv_7x7 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu'
    )(X)

    max_pool_1 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_7x7)

    conv_1x1 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu'
    )(max_pool_1)

    conv_3x3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        activation='relu'
    )(conv_1x1)

    max_pool_2 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(conv_3x3)

    inc_3a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    inc_3b = inception_block(inc_3a, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inc_3b)

    inc_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    inc_4b = inception_block(inc_4a, [160, 112, 224, 24, 64, 64])
    inc_4c = inception_block(inc_4b, [128, 128, 256, 24, 64, 64])
    inc_4d = inception_block(inc_4c, [112, 144, 288, 32, 64, 64])
    inc_4e = inception_block(inc_4d, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(inc_4e)

    inc_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    inc_5b = inception_block(inc_5a, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(inc_5b)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    softmax = K.layers.Dense(
        units=1000,
        activation='softmax'
    )(dropout)

    model = K.models.Model(inputs=X, outputs=softmax)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model
