# Notebooks de AirMed

Esta carpeta contiene los notebooks de Jupyter desarrollados
para el proyecto AirMed, orientado a la predicción de la
concentración de PM2.5 en Medellín mediante Deep Learning.

## Archivos

### 01_analisis_exploratorio.ipynb

Contiene el análisis exploratorio de los datos de SIATA,
incluyendo la revisión de calidad, los valores faltantes,
las concentraciones extremas y los patrones temporales
de PM2.5.

### 02_preprocesamiento.ipynb

Contiene el preprocesamiento de los datos, la construcción
de secuencias temporales y el entrenamiento y evaluación
de las arquitecturas LSTM y GRU.

También incluye la comparación de los modelos con una
referencia de persistencia y las métricas de evaluación.

## Entorno de ejecución

Los notebooks fueron desarrollados en Google Colab
utilizando Python y TensorFlow/Keras.

Para ejecutarlos se deben obtener los datos históricos
de SIATA y ajustar las rutas de los archivos según
la ubicación utilizada en Google Drive.
