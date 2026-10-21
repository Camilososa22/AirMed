"""Evalúa el modelo GRU guardado sobre el conjunto de prueba."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from preprocessing import prepare_data  # noqa: E402


def inverse_scale(values: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE (µg/m³)": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "RMSE (µg/m³)": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa AirMed GRU en el conjunto de prueba.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "airmed_medpblq_2026_01_08.csv",
        help="Ruta al CSV consolidado de SIATA.",
    )
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    model_path = ROOT_DIR / "models" / "airmed_gru_univariate.keras"
    scaler_path = ROOT_DIR / "models" / "robust_scaler_pm25.joblib"
    metrics_dir = ROOT_DIR / "results" / "metrics"
    figures_dir = ROOT_DIR / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo GRU: {model_path}. Ejecuta primero src/train.py.")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"No se encontró el escalador: {scaler_path}. Ejecuta primero src/train.py.")

    scaler = joblib.load(scaler_path)
    data = prepare_data(
        args.data,
        lookback=args.lookback,
        horizon=args.horizon,
        scaler=scaler,
    )
    test = data["test"]

    model = tf.keras.models.load_model(model_path)
    prediction_scaled = model.predict(test.X, verbose=0)
    y_test_real = inverse_scale(test.y, scaler)
    prediction_real = inverse_scale(prediction_scaled, scaler)

    # Referencia de persistencia: repetir la última observación de entrada.
    persistence_scaled = np.repeat(test.X[:, -1, 0, None], args.horizon, axis=1)
    persistence_real = inverse_scale(persistence_scaled, scaler)

    results = pd.DataFrame(
        [
            {"Modelo": "Persistencia", **metrics(y_test_real, persistence_real)},
            {"Modelo": "GRU", **metrics(y_test_real, prediction_real)},
        ]
    )

    results.to_csv(metrics_dir / "metricas_finales_prueba.csv", index=False)
    print("Resultados del conjunto de prueba:")
    print(results.round(3).to_string(index=False))

    # Guardar un ejemplo de pronóstico de 24 horas.
    import matplotlib.pyplot as plt

    sample = 0
    start = int(test.target_indices[sample])
    dates = data["df"]["timestamp"].iloc[start : start + args.horizon]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_test_real[sample], marker="o", label="PM2.5 observado")
    ax.plot(dates, prediction_real[sample], marker="s", label="PM2.5 predicho por GRU")
    ax.set_title("AirMed: ejemplo de predicción de PM2.5 en el conjunto de prueba")
    ax.set_xlabel("Fecha y hora")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "prediccion_gru_prueba.png", dpi=150)
    plt.close(fig)

    print(f"Métricas guardadas en: {metrics_dir / 'metricas_finales_prueba.csv'}")
    print(f"Gráfica guardada en: {figures_dir / 'prediccion_gru_prueba.png'}")


if __name__ == "__main__":
    main()
