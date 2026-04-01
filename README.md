# FinDataMining 📈🤖

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)

Proyecto integral de minería de datos y Machine Learning aplicado al sector financiero. El objetivo principal es predecir la capitalización de mercado (`MarketCap`) de los constituyentes del S&P 500 en función de una serie de métricas y ratios fundamentales, utilizando datos extraídos directamente de Yahoo Finance.

## 🗂️ Estructura del Proyecto

El flujo de trabajo (Pipeline) se divide en tres fases principales contenidas en Jupyter Notebooks, complementadas por un script modular de funciones:

* **`01_Extract.ipynb`**: Extracción y descarga de datos a través de la API de Yahoo Finance (`yfinance`). Se recuperan ratios fundamentales y precios, exportándolos eficientemente en un archivo `raw_data.parquet`.
* **`02_Transform.ipynb`**: Análisis Exploratorio (EDA) y preprocesamiento de variables. Incluye tratamiento de valores nulos, corrección de asimetría (*skewness*) con transformaciones Logarítmicas/Yeo-Johnson, y manejo estadístico de outliers (Winsorization). Se guarda la salida en `clean_data.parquet`.
* **`03_Modeling.ipynb`**: Construcción y evaluación de modelos predictivos. 
  * Inferencia e importancia de features a través de una Regresión OLS (`statsmodels`).
  * Implementación de un modelo final de ensamblado (`RandomForestRegressor`) usando `Pipelines` y preprocesadores integrados de `scikit-learn`.
* **`funciones.py`**: Scripts de ayuda conteniendo las tareas de ETL, gestión gráfica y cálculos estadísticos (e.g. V de Cramer customizada, gestión de outliers).

## 📊 Dataset

Los datos son recolectados en tiempo real. Parten de la lista de símbolos de `constituents.csv` (S&P 500) de los cuales se extraen (entre otros):
- Beta, Dividend Yield, Forward PE, Trailing PEG Ratio, Enterprise To Ebitda.
- Return On Assets, Return on Equity, Debt to Equity, Revenue Growth, etc.

## 🚀 Requisitos e Instalación

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/sebakremis/FinDataMining.git](https://github.com/sebakremis/FinDataMining.git)
   cd FinDataMining
