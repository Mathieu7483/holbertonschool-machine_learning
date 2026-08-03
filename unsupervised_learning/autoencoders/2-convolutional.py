#!/usr/bin/env python3
"""Write a function that creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder
    Args:
        input_dims: tuple of integers containing the dimensions of the model
        input
        filters: list containing the number of filters for each convolutional
        layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the latent
        space representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    # --- ENCODER ---
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(inputs, x, name='encoder')

    # --- DECODER ---
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    rev_filters = list(reversed(filters))

    for i in range(len(rev_filters)):
        f = rev_filters[i]

        if i == len(rev_filters) - 1:
            pad = 'valid'
        else:
            pad = 'same'

        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding=pad)(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    outputs = keras.layers.Conv2D(input_dims[2], (3, 3), activation='sigmoid',
                                  padding='same')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # --- AUTOENCODER ---
    auto_input = inputs
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(auto_input, decoded_output, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
