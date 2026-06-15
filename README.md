# 🌲 FinDataMining

### Pipeline automatizado de preparación de datos y modelado predictivo para ratios financieros del S&P 500.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)
![UCM](https://img.shields.io/badge/UCM-Máster_Data_Science-8b0000.svg)

---

Este proyecto implementa un pipeline ETL (Extracción, Transformación y Carga) automatizado para construir un dataset financiero exhaustivo a partir de las acciones constituyentes del índice S&P 500. El flujo utiliza fuentes de datos de acceso libre y gratuito, principalmente a través de la librería `yfinance`, permitiendo calcular métricas y ratios financieros clave basados en balances corporativos e históricos de precios. Una vez procesado, normalizado y limpio, el dataset se utiliza en la fase de modelado predictivo para entrenar algoritmos de Machine Learning orientados a la estimación de precios.

## 🚧 Estado del Proyecto

Este proyecto se encuentra actualmente en **fase activa de desarrollo**:
* **Etapa actual:** Los Jupyter Notebooks provistos están estructurados específicamente para ser ejecutados celda a celda. Este diseño interactivo facilita el análisis paso a paso, la experimentación matemática, el diagnóstico visual del pipeline y la calibración fina de los modelos de Machine Learning.
* **Evolución planificada:** En las próximas iteraciones, el flujo de trabajo será refactorizado y modularizado por completo en scripts de Python puro (`.py`). Asimismo, se incorporará una interfaz gráfica avanzada y un panel de control interactivo desarrollado en **Streamlit**, permitiendo la gestión automatizada del pipeline y la visualización dinámica de las métricas y predicciones.

## 🗂️ Estructura Actual del Repositorio

El flujo de trabajo está modularizado en tres fases principales desarrolladas en Jupyter Notebooks, las cuales se apoyan en un script de funciones complementarias:

```text
FINDATAMINING/
├── data/                       # Almacenamiento local de datasets y archivos de configuración
│   ├── resultados/             # Sub-directorio para almacenar los resultados de la aplicación de modelos
│   ├── constituents.csv        # Fichero de las acciones constituyentes del Indice S&P 500
│   ├── raw_data.parquet        # Datos crudos generados por la fase de Extracción
│   ├── clena_data.parquet      # Datos limpios generados en la fase Transform
├── src/                        # Sub-directorio con los módulos de funciones auxiliares
│   ├── __init__.py             # Fichero vacío, inicializa la carpeta como paquete
│   ├── config.py               # Configuración global del proyecto #data_sources.example.py
│   ├── data_sources.example.py # Fichero ejemplo para gestionar la clave API de FRED (ya no se utiliza en la versión actual) 
│   ├── evaluators.py           # Funciones auxiliares reutilizables para las tareas de la fase `Modeling`
│   ├── ingestion.py            # Funciones para la fase `Extract`
│   └── preprocessing.py        # Funciones para la fase `Transform`
├── .gitignore                  # Reglas de git ignore
├── 01_Extract.ipynb            # Cálculo de ratios fundamentales y exporta los resultados en formato parquet
├── 02_Transform.ipynb          # Análisis Exploratorio de Datos (EDA) y preprocesamiento avanzado de variables
├── 03_Modeling.ipynb           # Construcción, optimización y evaluación de modelos predictivos
├── LICENSE                     # Licencia del proyecto
├── README.md                   # Descripción del proyecto
└── requirements.txt            # Dependencias necesarias
```

## 📊 Dataset y Variables

El universo de datos se define a partir de los componentes oficiales del S&P 500 provistos en `constituents.csv`. Tras cruzar la información de los estados financieros con las series de precios históricos, se estructuran las siguientes dimensiones:

* **Variables explicativas (Features):** Métricas operativas, de riesgo y estructura de capital, tales como `Return On Assets` (ROA), `Return on Equity` (ROE), `Debt to EBITDA`, `Profit Margins`, entre otras.
* **Variable objetivo (Target):** El logaritmo del precio de cierre mensual.


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

## 📜 Licencia
Este repositorio se encuentra bajo una licencia de código abierto **MIT**. 
Puedes usar, modificar y distribuir el código libremente para cualquier propósito.

Para obtener más información, consulta el archivo [LICENCIA MIT](LICENSE).

⚠️ **Importante:** Este proyecto se desarrolló estrictamente con fines **académicos y educativos**.
`FinDataMining` **no** proporciona asesoramiento financiero ni recomendaciones de inversión. Los usuarios son los únicos responsables de las decisiones que pudiesen tomar basándose en la información generada por el mismo.