# src/models/dl_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_sequences(df: pd.DataFrame,
                     target_col: str,
                     feature_cols: list,
                     lookback: int,
                     horizon: int = 1):
    """
    Transforme un DataFrame en séquences X, y pour modèles séquentiels.
    lookback : nombre de pas dans le passé
    horizon  : nombre de pas à prédire dans le futur
    Retourne X (n_samples, lookback, n_features) et y (n_samples, horizon)
    """
    if len(df) < lookback + horizon:
        raise ValueError("Le DataFrame est trop court pour le lookback et horizon spécifiés.")
    data = df[feature_cols].values
    target = df[target_col].values
    n_samples = len(df) - lookback - horizon + 1
    X = np.zeros((n_samples, lookback, len(feature_cols)))
    y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X[i] = data[i:i+lookback]
        y[i] = target[i+lookback:i+lookback+horizon]
    return X, y

def scale_data(train: np.ndarray,
               test: np.ndarray):
    """
    Scale features via StandardScaler sur la première dimension.
    Retourne train_scaled, test_scaled, scaler
    """
    ns, lb, nf = train.shape
    train_flat = train.reshape(-1, nf)
    scaler = StandardScaler().fit(train_flat)
    train_scaled = scaler.transform(train_flat).reshape(ns, lb, nf)
    # pour test
    ms = test.shape[0]
    test_flat = test.reshape(-1, nf)
    test_scaled = scaler.transform(test_flat).reshape(ms, lb, nf)
    return train_scaled, test_scaled, scaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, TimeDistributed, Reshape

def build_lstm(input_shape, horizon=1, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['rmse'])
    return model

def build_gru(input_shape, horizon=1, units=64, dropout=0.2):
    model = Sequential([
        GRU(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['rmse'])
    return model

def build_cnn(input_shape, horizon=1, filters1=32, filters2=64, kernel_size=3, dropout=0.2):
    model = Sequential([
        Conv1D(filters1, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters2, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropout),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['rmse'])
    return model

def build_cnn_lstm(input_shape, horizon=1, n_subseq=4, filters=32, kernel_size=3, dropout=0.2, units=64):
    """
    CNN-LSTM :
    - on découpe la séquence lookback en n_subseq sous-séquences
    - applique un Conv1D+Pooling sur chacune (TimeDistributed)
    - on flattens et on passe tout ça dans un LSTM final
    input_shape = (lookback, n_features)
    """
    lookback, n_features = input_shape
    if lookback % n_subseq != 0:
        raise ValueError(f"lookback={lookback} n'est pas divisible par n_subseq={n_subseq}")
    subseq_len = lookback // n_subseq
    model = Sequential([
        Reshape((n_subseq, subseq_len, n_features), input_shape=input_shape),
        TimeDistributed(Conv1D(filters, kernel_size=kernel_size, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        LSTM(units),
        Dropout(dropout),
        Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['rmse'])
    return model
