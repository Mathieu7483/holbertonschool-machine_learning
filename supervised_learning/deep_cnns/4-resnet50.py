#!/usr/bin/env python3
"""Write a function that builds the ResNet-50 architecture
as described in Deep Residual Learning for Image
Recognition (2015):"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture as described in Deep Residual Learning
    for Image Recognition (2015):
    The model should be compiled using Adam optimization and categorical
    cross-entropy loss, and should have accuracy metrics
    Returns: the ResNet-50 model
    """
    X = K.Input(shape=(224, 224, 3))
    X = K.layers.ZeroPadding2D(padding=(3, 3))(X)

    # Stage 1
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                        padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    # Stage 2
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=K.initializers.
                       he_normal(seed=0))(X)
    model = K.models.Model(inputs=X, outputs=X)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
