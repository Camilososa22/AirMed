# Scripts Python de AirMed

Esta carpeta contiene los scripts Python reutilizables desarrollados para el proyecto AirMed, cuyo objetivo es predecir la concentración de material particulado PM2.5 en Medellín mediante técnicas de Deep Learning.

Los scripts permiten organizar las principales tareas del proyecto sin depender exclusivamente de los notebooks de Google Colab.

## Archivos

### `preprocessing.py`

Contiene las funciones para preparar los datos históricos de PM2.5, verificar la continuidad de las mediciones, tratar los valores faltantes y construir las secuencias temporales utilizadas por las redes neuronales.

### `models.py`

Define las arquitecturas de Deep Learning utilizadas en el proyecto:

- LSTM (Long Short-Term Memory).
- GRU (Gated Recurrent Unit).

Ambas arquitecturas están implementadas mediante TensorFlow/Keras y están diseñadas para utilizar 168 horas de datos históricos y generar 24 predicciones horarias.

### `train.py`

Contiene el procedimiento de entrenamiento y validación de los modelos LSTM y GRU.

Incluye la configuración de los experimentos, el seguimiento de las métricas, los mecanismos de control del entrenamiento y el guardado de los modelos y resultados obtenidos.

### `evaluate.py`

Permite evaluar el modelo GRU entrenado utilizando el conjunto de prueba y calcular métricas de desempeño como MAE y RMSE.

También permite comparar sus resultados con un modelo de referencia de persistencia.

## Requisitos

Para ejecutar los scripts se necesita Python y las dependencias definidas en el archivo `requirements.txt`, ubicado en la raíz del repositorio.

Las principales bibliotecas utilizadas son:

- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Joblib

## Ejecución

Los siguientes comandos deben ejecutarse desde la raíz del repositorio AirMed, después de instalar las dependencias y preparar el conjunto de datos.

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
