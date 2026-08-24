#!/usr/bin/env python3
"""
Module for preprocessing BTC historical datasets.
Cleans missing values, resamples data to 1-hour intervals, normalizes features,
and saves the dataset for training an RNN model.
"""
import numpy as np
import pandas as pd


def preprocess(file_path):
    """
    Preprocesses raw BTC CSV dataset.

    Args:
        file_path (str): Path to the raw CSV dataset file.

    Returns:
        tuple: (X_train, y_train, X_val, y_val) preprocessed arrays.
    """
    # 1. Chargement et conversion temporelle
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
    df.set_index('Timestamp', inplace=True)

    # 2. Nettoyage des trous de cotation (forward fill pour les prix)
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Weighted_Price'] = df['Weighted_Price'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    # 3. Aggrégation par heure (24 timesteps = 24 heures)
    df_hourly = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }).dropna()

    # Sélection des features (alignement PEP 8 sur 4 espaces)
    selected_cols = [
        'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price'
    ]
    data = df_hourly[selected_cols].values

    # 4. Normalisation Min-Max (calculée uniquement sur le Train)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    min_val = np.min(train_data, axis=0)
    max_val = np.max(train_data, axis=0)
    # Éviter la division par zéro
    range_val = np.where((max_val - min_val) == 0, 1, max_val - min_val)

    train_scaled = (train_data - min_val) / range_val
    val_scaled = (val_data - min_val) / range_val

    # Save to file
    np.savez("preprocessed_btc.npz", train=train_scaled, val=val_scaled)
    return train_scaled, val_scaled


if __name__ == "__main__":
    preprocess("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
