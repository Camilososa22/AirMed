# Modelos entrenados

Esta carpeta contiene los artefactos del modelo predictivo AirMed.

- `airmed_gru_univariate.keras`: red neuronal GRU entrenada para predecir PM2.5.
- `robust_scaler_pm25.joblib`: escalador utilizado durante el preprocesamiento.
- `configuracion_experimento.json`: parámetros y configuración del experimento.

El modelo utiliza una secuencia histórica de 168 horas para predecir las siguientes 24 horas.
