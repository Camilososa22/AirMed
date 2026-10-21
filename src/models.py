"""Definiciones de las arquitecturas recurrentes de AirMed."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, Input
from tensorflow.keras.optimizers import Adam


def build_lstm(
    lookback: int = 168,
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.20,
    horizon: int = 24,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Construye la arquitectura LSTM usada en el experimento."""
    model = Sequential(
        [
            Input(shape=(lookback, 1)),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=dense_units, activation="relu"),
            Dense(units=horizon, activation="linear"),
        ],
        name="airmed_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_gru(
    lookback: int = 168,
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.20,
    horizon: int = 24,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Construye la arquitectura GRU usada en el experimento."""
    model = Sequential(
        [
            Input(shape=(lookback, 1)),
            GRU(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=dense_units, activation="relu"),
            Dense(units=horizon, activation="linear"),
        ],
        name="airmed_gru",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
