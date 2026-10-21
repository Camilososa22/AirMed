"""Carga, limpieza ligera y construcción de secuencias para AirMed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "pm25_ug_m3"
MISSING_FLAG_COLUMN = "pm25_was_missing"


@dataclass
class SequenceSplit:
    """Secuencias de entrada, objetivos y sus índices temporales."""

    X: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Carga y valida la serie horaria de PM2.5.

    Los valores faltantes se marcan antes de imputar. Se utiliza forward fill
    con un máximo de tres horas consecutivas, igual que en el experimento.
    Los valores imputados nunca se usan como objetivos de entrenamiento o test.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {path}\n"
            "Descarga el CSV consolidado de SIATA y pásalo con --data."
        )

    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    required = {TIMESTAMP_COLUMN, TARGET_COLUMN}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Al dataset le faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    df = df[[TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN], errors="coerce"
    )
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Hay fechas vacías o inválidas en la columna timestamp.")

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if df[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError("Se encontraron marcas de tiempo duplicadas.")

    intervals = df[TIMESTAMP_COLUMN].diff().dropna()
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "La serie no es estrictamente horaria. Revisa los intervalos "
            "entre timestamps antes de crear secuencias."
        )

    # Conservar una marca de qué valores eran realmente observados.
    df[MISSING_FLAG_COLUMN] = df[TARGET_COLUMN].isna()

    # Imputación causal y limitada; nunca se rellenan más de 3 horas seguidas.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].ffill(limit=3)

    if df[TARGET_COLUMN].isna().any():
        count = int(df[TARGET_COLUMN].isna().sum())
        raise ValueError(
            f"Quedan {count} valores PM2.5 sin completar. Puede haber valores "
            "faltantes al inicio o huecos mayores a 3 horas; revísalos antes de entrenar."
        )

    return df


def _make_split(
    scaled_values: np.ndarray,
    observed_mask: np.ndarray,
    start_target: int,
    end_target_exclusive: int,
    lookback: int,
    horizon: int,
) -> SequenceSplit:
    """Crea ventanas cuyos objetivos pertenecen al intervalo indicado."""
    X_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    target_indices: list[int] = []

    first_target = max(start_target, lookback)
    last_target_start = end_target_exclusive - horizon

    for target_start in range(first_target, last_target_start + 1):
        target_end = target_start + horizon

        # No evaluar ni entrenar con etiquetas imputadas.
        if not observed_mask[target_start:target_end].all():
            continue

        X_values.append(
            scaled_values[target_start - lookback : target_start].reshape(
                lookback, 1
            )
        )
        y_values.append(scaled_values[target_start:target_end])
        target_indices.append(target_start)

    if not X_values:
        return SequenceSplit(
            X=np.empty((0, lookback, 1), dtype=np.float32),
            y=np.empty((0, horizon), dtype=np.float32),
            target_indices=np.empty((0,), dtype=np.int64),
        )

    return SequenceSplit(
        X=np.asarray(X_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.float32),
        target_indices=np.asarray(target_indices, dtype=np.int64),
    )


def prepare_data(
    csv_path: str | Path,
    lookback: int = 168,
    horizon: int = 24,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    scaler: Optional[RobustScaler] = None,
) -> dict:
    """Prepara divisiones temporales y secuencias para AirMed.

    Si no se pasa escalador, se ajusta solo con observaciones reales de PM2.5
    en la porción de entrenamiento. Si se pasa un escalador ya ajustado, se
    reutiliza sin volver a ajustarlo (útil para evaluación reproducible).
    """
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback y horizon deben ser enteros positivos.")
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction debe estar entre 0 y 1.")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction debe estar entre 0 y 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "La suma de las fracciones de entrenamiento y validación debe ser menor que 1."
        )

    df = load_dataset(csv_path)
    n_rows = len(df)
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))

    if train_end <= lookback + horizon:
        raise ValueError("No hay suficientes registros para la configuración temporal elegida.")

    observed_mask = (~df[MISSING_FLAG_COLUMN]).to_numpy(dtype=bool)

    if scaler is None:
        observed_training_values = df.loc[
            : train_end - 1,
            [TARGET_COLUMN, MISSING_FLAG_COLUMN],
        ]
        observed_training_values = observed_training_values.loc[
            ~observed_training_values[MISSING_FLAG_COLUMN], [TARGET_COLUMN]
        ]
        if observed_training_values.empty:
            raise ValueError("No hay mediciones reales de PM2.5 para ajustar el escalador.")
        scaler = RobustScaler()
        scaler.fit(observed_training_values[[TARGET_COLUMN]])

    scaled_values = scaler.transform(df[[TARGET_COLUMN]]).astype(np.float32).reshape(-1)

    train = _make_split(
        scaled_values, observed_mask, 0, train_end, lookback, horizon
    )
    validation = _make_split(
        scaled_values, observed_mask, train_end, validation_end, lookback, horizon
    )
    test = _make_split(
        scaled_values, observed_mask, validation_end, n_rows, lookback, horizon
    )

    if min(len(train.X), len(validation.X), len(test.X)) == 0:
        raise ValueError(
            "Uno de los conjuntos no tiene suficientes secuencias completas. "
            "Revisa el período o reduce lookback/horizon."
        )

    return {
        "df": df,
        "scaler": scaler,
        "scaled_values": scaled_values,
        "observed_mask": observed_mask,
        "train": train,
        "validation": validation,
        "test": test,
        "train_end": train_end,
        "validation_end": validation_end,
        "lookback": lookback,
        "horizon": horizon,
    }
