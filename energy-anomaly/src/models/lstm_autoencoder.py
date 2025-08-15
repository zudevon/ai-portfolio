#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def build_lstm_autoencoder(seq_len: int, n_features: int, lstm_units: int = 64, latent_dim: int = 32, dropout: float = 0.1, l2: float = 0.0) -> Model:
    """
    LSTM Autoencoder for multivariate sequences.
    Input: (batch, seq_len, n_features) -> Output: same shape
    """
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None

    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Masking()(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(latent_dim, return_sequences=False, kernel_regularizer=reg)(x)
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=reg)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model
