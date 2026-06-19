# 📊 FinDataMining

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)
![UCM](https://img.shields.io/badge/UCM-Máster_Data_Science-8b0000.svg)

Este proyecto implementa un pipeline ETL (Extracción, Transformación y Carga) para construir un dataset financiero sobre las acciones constituyentes del índice S&P 500 procedentes de fuentes de datos gratuitas. Una vez procesados, normalizados y limpios, los datos son utilizados en la fase de modelado predictivo para entrenar algoritmos de Machine Learning.

El objetivo es proporcionar un entorno de experimentación ágil para científicos de datos. A modo de validación, el proyecto implementa un modelo base de *RandomForest*, el cual arroja métricas de ajuste moderadas, un resultado previsible dada la naturaleza ruidosa y no estacionaria de los datos financieros, asi como la disponibilidad limitada de datos. De este modo, el repositorio queda preparado para iterar y probar fácilmente otros algoritmos de Machine Learning tradicional o bien modelos de redes neuronales.

## 🗄️ Fuentes de Datos

El pipeline de extracción obtiene los datos históricos de precios de mercado y los balances corporativos de forma libre y gratuita, a través de las librerías: 

- `yfinance`: Se obtienen los precios históricos, asi como datos financieros correspondientes a los últimos cuatro reportes trimestrales. 

- `simFin`: A través de una cuenta gratuita, `simFin` ofrece datos trimestrales de 5 años, con un año de retraso. Su uso requiere de una clave API, la cual se obtiene registrándose en el sitio web (https://simfin.com/). Las instrucciones para ingresar la clave se encuentran en el fichero src/data_sources.example.py.
Otra restricción de la cuenta básica es que no ofrece información para todos los tickers. Esto obliga a restringir el universo de tickers en el proyecto, reduciendo la cantidad de tickers en el dataset a 384 de los 500 componentes del índice.

## 🚧 Estado del Proyecto

`finDataMining` se encuentra en **fase activa de desarrollo**:
* **Etapa actual:** Los Jupyter Notebooks provistos están estructurados específicamente para ser ejecutados celda a celda. Este diseño interactivo facilita el análisis paso a paso, la experimentación matemática, el diagnóstico visual del pipeline y la calibración de los modelos de Machine Learning.
* **Evolución planificada:** Se incorporará un panel de control interactivo desarrollado en **Streamlit**, permitiendo la gestión automatizada del pipeline sin la necesidad de utilizar los Notebooks, asi como la visualización dinámica de las métricas y predicciones. Al integrar el flujo en aplicaciones de tipo `.py`, se desarrollarán características para poder actualizar y gestionar la base de datos.

## 🗂️ Estructura Actual del Repositorio

El flujo de trabajo está modularizado en tres fases principales desarrolladas en Jupyter Notebooks, las cuales se apoyan en un script de funciones complementarias:

```text
FINDATAMINING/
├── data/                            # Almacenamiento local de datasets y archivos de configuración
│   ├── reports/                     # Sub-directorio para almacenar los reportes generados luego de modelar
│   ├── simfin/                      # Almacena los ficheros de datos de simFin
│   ├── constituents.csv             # Fichero de las acciones constituyentes del Indice S&P 500
│   ├── raw_data.parquet             # Datos crudos generados por la fase de Extracción
│   ├── clean_data.parquet           # Datos limpios generados en la fase Transform
│   ├── market_index.parquet         # Datos históricos de precios del índice del mercado
│   ├── tickers_universe.csv         # Se guardan los tickers sobre los cuales exiten datos
├── src/                             # Sub-directorio con los módulos de funciones auxiliares
│   ├── __init__.py                  # Fichero vacío, inicializa la carpeta como paquete
│   ├── config.py                    # Configuración global del proyecto #data_sources.example.py
│   ├── data_sources.example.py      # Fichero ejemplo para gestionar la clave API de simFin
│   ├── extract.py                   # Módulo de la fase de extracción
│   ├── modeling.py                  # Módulo de la fase de modelado
│   └── transform.py                 # Módulo de la fase transformación
├── .gitignore                       # Reglas de git ignore
├── 01_Notebook_Extraccion.ipynb     # Cálculo de ratios fundamentales y exporta los resultados en formato parquet
├── 02_Notebook_Transformacion.ipynb # Análisis Exploratorio de Datos (EDA) y preprocesamiento avanzado de variables
├── 03_Notebook_Modelado.ipynb       # Construcción, optimización y evaluación de modelos predictivos
├── LICENSE                          # Licencia del proyecto
├── README.md                        # Descripción del proyecto
└── requirements.txt                 # Dependencias necesarias
```

## 📊 Dataset y Variables

Tras cruzar la información de los estados financieros con las series de precios históricos, se estructuran las siguientes dimensiones:

* **Variables explicativas (Features):** Métricas operativas, de riesgo y estructura de capital, tales como `Return On Assets` (ROA), `Return on Equity` (ROE), `Debt to EBITDA`, `Profit Margins`, entre otras.
* **Variable objetivo (Target):** La fase de modelado permite seleccionar y experimentar con distintas variables objetivo, tales como los precios trimestrales, la Capitalización Bursátil o alguno de los ratios de valuación que se incluyen en el dataset.

## 🚀 Requisitos e Instalación

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/sebakremis/FinDataMining.git
   cd FinDataMining
   ```

* Antes de instalar las librerías, se recomienda crear un entorno virtual específico para trabajar con *shap*, porque requiere de una versión antigüa de *numpy*.

2. Instala el entorno de dependencias requerido utilizando `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## 📜 Licencia
Este repositorio se encuentra bajo una licencia de código abierto **MIT**. 
Puedes usar, modificar y distribuir el código libremente para cualquier propósito.

Para obtener más información, consulta el archivo [LICENCIA MIT](LICENSE).

⚠️ **Importante:** Este proyecto se desarrolló estrictamente con fines **académicos y educativos**.
`FinDataMining` **no** proporciona asesoramiento financiero ni recomendaciones de inversión. Los usuarios son los únicos responsables de las decisiones que pudiesen tomar basándose en la información generada por el mismo.
