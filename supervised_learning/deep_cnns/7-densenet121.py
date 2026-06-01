#!/usr/bin/env python3
"""Write a function that builds the DenseNet-121 architecture
as described in Densely Connected Convolutional Networks:"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks.

    Args:
        growth_rate (int): the growth rate for the dense blocks.
        compression (float): the compression factor for the transition layers.

    Returns:
        keras.Model: the DenseNet-121 model.
    """
    # Input layer
    X = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    Y = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    Y = K.layers.BatchNormalization(axis=3)(Y)
    Y = K.layers.Activation('relu')(Y)
    Y = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Y)

    # Dense Block 1
    Y, nb_filters = dense_block(Y, 64, growth_rate, 6)

    # Transition Layer 1
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    # Dense Block 2
    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 12)

    # Transition Layer 2
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    # Dense Block 3
    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 24)

    # Transition Layer 3
    Y, nb_filters = transition_layer(Y, nb_filters, compression)

    # Dense Block 4
    Y, nb_filters = dense_block(Y, nb_filters, growth_rate, 16)

    # Global Average Pooling and Output layer
    Y = K.layers.GlobalAveragePooling2D()(Y)
    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=K.initializers.he_normal(seed=0)
                       )(Y)

    return K.models.Model(inputs=X, outputs=Y)
