# Dataset de AirMed

Esta carpeta contiene el conjunto de datos consolidado
utilizado para desarrollar el proyecto AirMed.

## Archivo

`airmed_medpblq_2026_01_08.csv`

## Características

- Estación: MED-PBLQ.
- Ubicación: Medellín, Colombia.
- Período: enero a agosto de 2026.
- Frecuencia: horaria.
- Total de registros: 5.832.
- Variable objetivo: PM2.5.
- Unidad de medida: µg/m³.

## Fuente de información

Los datos se consolidaron a partir de los archivos mensuales
de datos validados del Sistema de Alerta Temprana de Medellín
y el Valle de Aburrá (SIATA).

Fuente oficial:
https://siata.gov.co/IDEAM/

## Preparación

El archivo conserva los valores faltantes originales.
La imputación de valores y la construcción de secuencias
temporales se realizan durante el preprocesamiento.

El modelo predictivo inicial utiliza únicamente la variable
PM2.5, aunque el CSV incluye otras variables ambientales.

## Uso

Este conjunto se utiliza con fines académicos y experimentales.
Se debe conservar la atribución de la fuente original y
consultar sus condiciones de uso y distribución.
