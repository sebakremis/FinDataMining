# FinDataMining 📈🤖

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)

Este proyecto implementa un pipeline ETL (Extracción, Transformación y Carga) automatizado para construir un dataset financiero exhaustivo a partir de las acciones constituyentes del índice S&P 500. El flujo utiliza fuentes de datos de acceso libre y gratuito, principalmente a través de la librería `yfinance`, permitiendo calcular métricas y ratios financieros clave basados en balances corporativos e históricos de precios. Una vez procesado, normalizado y limpio, el dataset se utiliza en la fase de modelado predictivo para entrenar algoritmos de Machine Learning orientados a la estimación de múltiples ratios de valuación.

## 🗂️ Estructura del Proyecto

El flujo de trabajo está modularizado en tres fases principales desarrolladas en Jupyter Notebooks, las cuales se apoyan en un script de funciones complementarias:

* **`01_Extract.ipynb`**: Extracción y descarga automatizada de datos desde la API de Yahoo Finance (`yfinance`). Realiza el cálculo inicial de ratios fundamentales y exporta los resultados en formato óptimo `raw_data.parquet`.
* **`02_Transform.ipynb`**: Análisis Exploratorio de Datos (EDA) y preprocesamiento avanzado de variables. Incluye la imputación y tratamiento de valores nulos, corrección de la asimetría (*skewness*) mediante transformaciones logarítmicas y de Yeo-Johnson, y la gestión estadística de valores atípicos (*outliers*) mediante *Winsorization*. Almacena la salida limpia en `clean_data.parquet`.
* **`03_Modeling.ipynb`**: Construcción, optimización y evaluación de modelos predictivos. 
  * Implementación de un modelo final basado en ensambles (`RandomForestRegressor`) estructurado mediante `Pipelines` y preprocesadores integrados de `scikit-learn`.
  * Análisis de inferencia y determinación de la importancia de las características (*feature importance*).
  * Aplicación del modelo entrenado para generar predicciones sobre las últimas observaciones de cada ticker.
* **`funciones.py`**: Script modular que contiene funciones auxiliares reutilizables para las tareas de ETL, generación de gráficos y cálculos estadísticos avanzados.

## 📊 Dataset y Variables

El universo de datos se define a partir de los componentes oficiales del S&P 500 provistos en `constituents.csv`. Tras cruzar la información de los estados financieros con las series de precios históricos, se estructuran las siguientes dimensiones:

* **Variables explicativas (Features):** Métricas operativas, de riesgo y estructura de capital, tales como `Beta`, `Return On Assets` (ROA), `Return on Equity` (ROE), `Debt to Equity`, `EnterpriseValue`, entre otras.
* **Variables objetivo (Targets):** Ratios de valuación de mercado analizados de forma independiente (`PriceToBook`, `PE_Trailing`, `EnterpriseToEbitda`), permitiendo configurar el pipeline para predecir cualquiera de ellos de manera paramétrica durante la fase de modelado.

## 🚀 Requisitos e Instalación

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/sebakremis/FinDataMining.git
   cd FinDataMining
   ```

2. Instala el entorno de dependencias requerido utilizando `pip`:
   ```bash
   pip install -r requirements.txt
   ```