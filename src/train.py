"""Entrena y compara las redes LSTM y GRU de AirMed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Permite ejecutar: python src/train.py desde la raíz del repositorio.
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from models import build_gru, build_lstm  # noqa: E402
from preprocessing import prepare_data  # noqa: E402


def inverse_scale(values: np.ndarray, scaler) -> np.ndarray:
    """Convierte matrices normalizadas a µg/m³."""
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula MAE y RMSE sobre todos los horizontes evaluados."""
    return {
        "MAE (µg/m³)": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "RMSE (µg/m³)": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos AirMed LSTM y GRU.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT_DIR / "data" / "raw" / "airmed_medpblq_2026_01_08.csv",
        help="Ruta al CSV consolidado de SIATA.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    model_dir = ROOT_DIR / "models"
    metrics_dir = ROOT_DIR / "results" / "metrics"
    figures_dir = ROOT_DIR / "results" / "figures"
    for directory in (model_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = prepare_data(
        args.data,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    train = data["train"]
    val = data["validation"]
    scaler = data["scaler"]

    print("AIRMED: DATOS PREPARADOS")
    print(f"Entrenamiento: X={train.X.shape}, y={train.y.shape}")
    print(f"Validación:    X={val.X.shape}, y={val.y.shape}")
    print(f"Prueba:        X={data['test'].X.shape}, y={data['test'].y.shape}")

    y_val_real = inverse_scale(val.y, scaler)
    baseline_scaled = np.repeat(val.X[:, -1, 0, None], args.horizon, axis=1)
    baseline_real = inverse_scale(baseline_scaled, scaler)

    val_rows = []
    baseline_metrics = calculate_metrics(y_val_real, baseline_real)
    val_rows.append({"Modelo": "Persistencia", **baseline_metrics})

    callback_factory = lambda: [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    architectures = {
        "LSTM": build_lstm,
        "GRU": build_gru,
    }

    for model_name, builder in architectures.items():
        # Se fija la semilla antes de cada modelo para resultados más reproducibles.
        tf.keras.utils.set_random_seed(args.seed)
        model = builder(lookback=args.lookback, horizon=args.horizon)
        print(f"\nEntrenando AirMed {model_name}...")
        history = model.fit(
            train.X,
            train.y,
            validation_data=(val.X, val.y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callback_factory(),
            shuffle=True,
            verbose=1,
        )

        model_path = model_dir / f"airmed_{model_name.lower()}_univariate.keras"
        model.save(model_path)

        # Guardar el historial del entrenamiento.
        pd.DataFrame(history.history).to_csv(
            metrics_dir / f"historial_entrenamiento_{model_name.lower()}.csv",
            index_label="epoch",
        )

        # Guardar las curvas de aprendizaje para evidencias/documentación.
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history.history["loss"], label="Entrenamiento")
        axes[0].plot(history.history["val_loss"], label="Validación")
        axes[0].set_title(f"{model_name}: pérdida")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("MSE normalizado")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].plot(history.history["mae"], label="Entrenamiento")
        axes[1].plot(history.history["val_mae"], label="Validación")
        axes[1].set_title(f"{model_name}: MAE")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("MAE normalizado")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"curvas_{model_name.lower()}.png", dpi=150)
        plt.close(fig)

        val_pred_scaled = model.predict(val.X, verbose=0)
        val_pred_real = inverse_scale(val_pred_scaled, scaler)
        val_metrics = calculate_metrics(y_val_real, val_pred_real)
        val_rows.append({"Modelo": model_name, **val_metrics})

        print(f"{model_name} guardado en: {model_path}")
        print(f"Mejor val_loss (normalizada): {min(history.history['val_loss']):.6f}")
        print(f"MAE validación (µg/m³): {val_metrics['MAE (µg/m³)']:.3f}")
        print(f"RMSE validación (µg/m³): {val_metrics['RMSE (µg/m³)']:.3f}")

    # El escalador corresponde solamente a observaciones reales del período de entrenamiento.
    scaler_path = model_dir / "robust_scaler_pm25.joblib"
    joblib.dump(scaler, scaler_path)

    results_df = pd.DataFrame(val_rows)
    results_path = metrics_dir / "comparacion_modelos_validacion.csv"
    results_df.to_csv(results_path, index=False)

    config = {
        "proyecto": "AirMed",
        "estacion": "MED-PBLQ",
        "variable_objetivo": "pm25_ug_m3",
        "frecuencia": "horaria",
        "longitud_entrada_horas": args.lookback,
        "horizonte_prediccion_horas": args.horizon,
        "modelos_entrenados": ["LSTM", "GRU"],
        "unidades_recurrentes": 64,
        "unidades_dense": 32,
        "dropout": 0.2,
        "batch_size": args.batch_size,
        "epocas_maximas": args.epochs,
        "semilla": args.seed,
        "escalador": "RobustScaler, ajustado con observaciones reales del entrenamiento",
        "division_temporal": {"entrenamiento": 0.70, "validacion": 0.15, "prueba": 0.15},
        "criterio_seleccion": "MAE en validación",
        "fuente_csv": str(args.data),
    }
    with (model_dir / "configuracion_experimento.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print("\nComparación de validación:")
    print(results_df.round(3).to_string(index=False))
    print(f"\nMétricas guardadas: {results_path}")
    print(f"Escalador guardado: {scaler_path}")


if __name__ == "__main__":
    main()
