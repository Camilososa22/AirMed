# AirMed: Predicción de la calidad del aire en Medellín mediante Deep Learning

## 1. Descripción del proyecto

AirMed es un proyecto académico de aprendizaje profundo orientado a la predicción de la concentración de material particulado fino PM2.5 en una estación de monitoreo de calidad del aire de Medellín, Colombia.

El proyecto utiliza registros históricos de calidad del aire de la estación MED-PBLQ, obtenidos a partir de información del Sistema de Alerta Temprana de Medellín y el Valle de Aburrá (SIATA).

Se implementan y comparan dos arquitecturas de redes neuronales recurrentes: LSTM (Long Short-Term Memory) y GRU (Gated Recurrent Unit). Además, se utiliza un modelo de persistencia como referencia para evaluar si las redes neuronales mejoran las predicciones.

El desarrollo incluye análisis exploratorio, preprocesamiento de datos, construcción de secuencias temporales, entrenamiento, evaluación y documentación de los resultados.

## 2. Problemática

La contaminación atmosférica es una problemática ambiental relevante para Medellín y el Valle de Aburrá. El material particulado PM2.5 es un contaminante de interés debido a su pequeño tamaño y a sus posibles efectos sobre la salud.

Las estaciones de monitoreo permiten conocer las concentraciones registradas, pero predecir sus valores futuros constituye un problema que requiere analizar patrones temporales y evaluar el comportamiento de modelos predictivos.

En este contexto, AirMed busca determinar si las redes neuronales recurrentes pueden aprender patrones de las mediciones históricas y mejorar las estimaciones frente a un método de referencia sencillo.

El proyecto tiene fines académicos y experimentales. No constituye un sistema oficial de pronóstico ni de alertas ambientales.

## 3. Objetivos

### Objetivo general

Desarrollar y evaluar modelos de Deep Learning basados en redes neuronales recurrentes para predecir la concentración horaria de PM2.5 durante las siguientes 24 horas, utilizando registros históricos de una estación de monitoreo de Medellín.

### Objetivos específicos

- Analizar los datos históricos de PM2.5 e identificar sus características temporales, valores faltantes y observaciones extremas.
- Preparar las secuencias temporales para el entrenamiento de redes neuronales.
- Implementar modelos LSTM y GRU mediante TensorFlow/Keras.
- Evaluar los modelos mediante MAE y RMSE.
- Comparar los resultados con un modelo de persistencia.
- Documentar la metodología, los resultados y las limitaciones del proyecto.

## 4. Datos utilizados

| Característica | Descripción |
|---|---|
| Fuente | Sistema de Alerta Temprana de Medellín y el Valle de Aburrá (SIATA) |
| Estación | MED-PBLQ |
| Período analizado | Enero a agosto de 2026 |
| Frecuencia | Horaria |
| Número inicial de registros | 5.832 |
| Variable objetivo | PM2.5 |
| Unidad de medida | µg/m³ |
| Valores faltantes originales de PM2.5 | 68 |
| Porcentaje de valores faltantes | 1,17 % |

Fuente oficial de datos: [SIATA](https://siata.gov.co/IDEAM/).

El análisis exploratorio también considera variables meteorológicas disponibles en los archivos históricos. Sin embargo, la primera versión de los modelos predictivos utiliza únicamente las mediciones de PM2.5.

### Preparación de los datos

Durante el preprocesamiento se realizaron las siguientes actividades:

1. Ordenamiento cronológico de las observaciones.
2. Identificación de valores faltantes y comprobación de la continuidad temporal.
3. Tratamiento inicial de los valores faltantes mediante arrastre del último valor disponible, con un límite de tres horas consecutivas.
4. Conservación de una marca para identificar las observaciones originalmente faltantes.
5. Normalización mediante RobustScaler, ajustado utilizando mediciones reales del conjunto de entrenamiento.
6. Construcción de secuencias temporales de entrada y salida.

Las secuencias cuyo horizonte objetivo contenía mediciones originalmente faltantes se excluyeron del entrenamiento y la evaluación.

Los datos se dividieron cronológicamente en entrenamiento (70 %), validación (15 %) y prueba (15 %), antes de construir las secuencias de cada conjunto.

## 5. Formulación del problema predictivo

AirMed aborda un problema de predicción de series temporales.

El modelo recibe una secuencia histórica de 168 mediciones horarias de PM2.5, equivalentes a siete días de observaciones, y genera 24 predicciones horarias para el período siguiente.

| Parámetro | Configuración |
|---|---|
| Tipo de problema | Regresión de series temporales |
| Variable de entrada | PM2.5 |
| Longitud de la secuencia histórica | 168 horas |
| Horizonte de predicción | 24 horas |
| Frecuencia de las predicciones | Horaria |
| Modelos | LSTM y GRU |
| Modelo de referencia | Persistencia |

El modelo de persistencia utiliza la última medición histórica disponible como predicción constante para las siguientes 24 horas.

## 6. Arquitecturas de Deep Learning

### 6.1. LSTM

Se implementó una red neuronal recurrente con la siguiente arquitectura:

- Entrada de 168 pasos temporales y una variable.
- Capa LSTM de 64 unidades.
- Capa Dropout con una tasa de 0,20.
- Capa densa de 32 neuronas con activación ReLU.
- Capa de salida de 24 neuronas para generar las predicciones.

La red tiene 19.768 parámetros entrenables.

### 6.2. GRU

Se implementó una arquitectura alternativa con la siguiente estructura:

- Entrada de 168 pasos temporales y una variable.
- Capa GRU de 64 unidades.
- Capa Dropout con una tasa de 0,20.
- Capa densa de 32 neuronas con activación ReLU.
- Capa de salida de 24 neuronas para generar las predicciones.

La red tiene 15.736 parámetros entrenables.

### 6.3. Configuración del entrenamiento

Los modelos se implementaron mediante TensorFlow/Keras y utilizaron inicialmente la siguiente configuración:

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Tasa de aprendizaje inicial | 0,001 |
| Función de pérdida | Error cuadrático medio (MSE) |
| Tamaño del lote | 32 |
| Máximo de épocas | 50 |
| Early Stopping | Paciencia de 8 épocas |
| Reducción adaptativa de la tasa de aprendizaje | ReduceLROnPlateau |
| Semilla de inicialización | 42 |

El entrenamiento se controló mediante la pérdida de validación y la restauración de los pesos correspondientes a la mejor época.

## 7. Métricas de evaluación

Se utilizaron dos métricas principales.

### MAE (Mean Absolute Error)

Representa el promedio de los errores absolutos entre las predicciones y los valores observados. Un valor menor indica un menor error absoluto promedio.

### RMSE (Root Mean Squared Error)

Corresponde a la raíz del error cuadrático medio. Penaliza en mayor medida los errores de gran magnitud.

Ambas métricas se expresan en µg/m³ y permiten interpretar los resultados en la escala original de PM2.5.

## 8. Resultados experimentales

### 8.1. Resultados de validación

| Modelo | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistencia | 6,764 | 9,081 |
| LSTM | 5,254 | 6,652 |
| GRU | 5,131 | 6,473 |

Durante la validación, la GRU obtuvo los mejores resultados entre los tres modelos evaluados.

Frente al modelo de persistencia, la GRU redujo el MAE aproximadamente un 24,14 % y el RMSE un 28,72 %.

### 8.2. Resultados de prueba

Después de seleccionar la GRU a partir de los resultados de validación, se evaluó su desempeño en el conjunto de prueba.

| Modelo | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|
| Persistencia | 6,935 | 8,995 |
| GRU | 5,008 | 6,292 |

En el conjunto de prueba, la GRU obtuvo un MAE de 5,008 µg/m³ y un RMSE de 6,292 µg/m³.

Frente al modelo de persistencia, esto representa una reducción aproximada del 27,79 % en el MAE y del 30,05 % en el RMSE.

Los resultados indican que la GRU superó al método de referencia en las métricas analizadas durante el período de prueba. Sin embargo, el modelo no reproduce perfectamente todas las variaciones bruscas de PM2.5.

### Interpretación de los resultados

La comparación muestra que, en los datos evaluados, las redes recurrentes pueden aprovechar patrones temporales para mejorar las predicciones frente al método de persistencia.

La GRU obtuvo mejores resultados que la LSTM en la validación inicial y utiliza menos parámetros entrenables.

Estas conclusiones corresponden al período y a la estación analizados. No garantizan el mismo desempeño en otros períodos, estaciones o condiciones ambientales.

## 9. Estructura del repositorio

El proyecto se organiza de la siguiente manera:

```text
AirMed/
├── README.md
├── .gitignore
└── notebooks/
    ├── README.md
    ├── 01_analisis_exploratorio.ipynb
    └── 02_preprocesamiento_y_modelado.ipynb
```

### Descripción de los notebooks

**01_analisis_exploratorio.ipynb**

Contiene la carga de los datos, el análisis de calidad, la investigación de valores extremos, las visualizaciones y el estudio de los patrones temporales de PM2.5.

**02_preprocesamiento_y_modelado.ipynb**

Contiene el tratamiento inicial de los datos faltantes, la normalización, la construcción de secuencias, el entrenamiento de las redes LSTM y GRU y la comparación de sus resultados frente al modelo de persistencia.

Los artefactos entrenados, las métricas exportadas y la presentación del proyecto se incorporarán al repositorio como parte de la organización final de la entrega.

## 10. Herramientas utilizadas

- Python.
- Google Colab.
- Pandas.
- NumPy.
- Scikit-learn.
- TensorFlow/Keras.
- Matplotlib.
- GitHub.

## 11. Ejecución del proyecto

Los notebooks se desarrollaron en Google Colab.

Para reproducir los experimentos:

1. Acceder al repositorio público de AirMed.
2. Abrir el notebook deseado en Google Colab.
3. Obtener los archivos históricos desde la fuente de datos de SIATA.
4. Preparar el archivo consolidado de acuerdo con el formato utilizado por el notebook.
5. Ajustar la ruta del archivo en Google Drive si es diferente de la configurada en el código.
6. Ejecutar las celdas en orden.

El notebook de modelado espera encontrar el archivo consolidado en la siguiente ruta de Google Drive:

`/content/drive/MyDrive/AirMed/data/raw/airmed_medpblq_2026_01_08.csv`

Las dependencias utilizadas incluyen Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib y Joblib.

Los resultados dependen de la versión de las bibliotecas, las condiciones de ejecución y los datos utilizados.

## 12. Limitaciones y trabajo futuro

El proyecto tiene las siguientes limitaciones:

- El período analizado comprende ocho meses de datos y no permite evaluar completamente la variabilidad entre años.
- El modelo utiliza una sola variable de entrada y no incorpora todavía información meteorológica.
- Las concentraciones extremas pueden ser difíciles de anticipar y requieren un análisis adicional.
- La evaluación corresponde a una sola estación de monitoreo.
- Las predicciones sucesivas de 24 horas utilizan horizontes solapados, por lo que las métricas agregadas incluyen errores de diferentes anticipaciones.

Como trabajo futuro se propone:

- Incorporar variables meteorológicas y evaluar si mejoran las predicciones.
- Ampliar el período histórico con registros adicionales.
- Analizar el desempeño por hora de anticipación.
- Investigar el comportamiento del modelo durante episodios de contaminación elevada.
- Comparar diferentes configuraciones de las arquitecturas LSTM y GRU.
- Preparar una demostración que permita visualizar predicciones a partir de nuevas mediciones.

## 13. Autoría

Proyecto desarrollado como parte de un taller académico de Deep Learning.

**Nombre del proyecto:** AirMed  
**Tema:** Predicción de la calidad del aire mediante redes neuronales recurrentes  
**Ubicación de estudio:** Medellín, Colombia
