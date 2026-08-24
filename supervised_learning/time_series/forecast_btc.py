#!/usr/bin/env python3
"""
Module for building, training, and validating an RNN model to forecast BTC.
Uses tf.data.Dataset for feeding windowed sequential data into Keras.
"""
import numpy as np
import tensorflow as tf


def create_dataset(data, window_size=24, batch_size=32, shuffle=True):
    """
    Creates a windowed tf.data.Dataset for RNN time series forecasting.

    Args:
        data (numpy.ndarray): Scaled time series dataset.
        window_size (int): Lookback period in hours (default 24).
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset buffer.

    Returns:
        tf.data.Dataset: Batched dataset containing (X, y) tuples.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # Fenêtre glissante : window_size pour X + 1 pour y
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))

    # X = 24 timesteps (toutes les features), y = prix 'Close' à t+1 (index 0)
    dataset = dataset.map(lambda w: (w[:-1, :], w[-1, 0]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_and_train_model():
    """
    Loads data, constructs Keras RNN model, trains with MSE loss and saves it.
    """
    # Loading dataset
    raw_data = np.load("preprocessed_btc.npz")
    train_data = raw_data['train']
    val_data = raw_data['val']

    window_size = 24
    batch_size = 64
    epochs = 20

    train_ds = create_dataset(train_data, window_size, batch_size, True)
    val_ds = create_dataset(val_data, window_size, batch_size, False)

    num_features = train_data.shape[1]

    # Architecture RNN / LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(window_size, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    model.save("btc_forecast_model.h5")


if __name__ == "__main__":
    build_and_train_model()
