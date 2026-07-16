# src/modeling.py
"""
Módulo de la fase de modelado
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, confusion_matrix, ConfusionMatrixDisplay
    )
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from src.config import reports_folder

class CrossSectionalScaler(BaseEstimator, TransformerMixin):
    """
    Escalador Transversal (Cross-Sectional Z-Score).
    Estandariza variables numéricas calculando la media y desviación estándar 
    agrupada de forma estricta por fecha y sector.
    """
    def __init__(self, groupby_cols=['Date', 'Sector'], features=None, min_samples=5):
        self.groupby_cols = groupby_cols
        self.features = features
        self.min_samples = min_samples

    def fit(self, X, y=None):
        # Definimos las features a usar durante el fit para tenerlas disponibles
        if self.features is None:
            self.features_out_ = [col for col in X.columns if col not in self.groupby_cols and pd.api.types.is_numeric_dtype(X[col])]
        else:
            self.features_out_ = self.features
        return self

    def transform(self, X):
        missing_cols = [col for col in self.groupby_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Las columnas de agrupación {missing_cols} no están en X.")

        def z_score(series):
            if len(series.dropna()) < self.min_samples:
                return pd.Series(0, index=series.index, dtype=float)
            
            std = series.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=series.index, dtype=float)
                
            return (series - series.mean()) / std

        # Aplicamos el Z-score
        scaled_features = X.groupby(self.groupby_cols, observed=True)[self.features_out_].transform(z_score)

        # Llenamos nulos con 0
        scaled_features = scaled_features.fillna(0)

        return scaled_features

    def get_feature_names_out(self, input_features=None):
        """
        Método requerido por Scikit-Learn para rastrear los nombres de las columnas 
        después de que pasan por el ColumnTransformer.
        """
        return np.array(self.features_out_)



def split_ultimo(
    df: pd.DataFrame, 
    label: str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha más reciente.
    """
    
    # Identificamos cuál es la fecha máxima en todo el dataset
    ultima_fecha = df['Date'].max()
    
    # Separamos mediante máscaras booleanas
    df_train = df[df['Date'] < ultima_fecha]
    df_test = df[df['Date'] == ultima_fecha]
    
    X_train = df_train.drop(columns=cols_excluded)
    y_train = df_train[label]
    
    X_test = df_test.drop(columns=cols_excluded)
    y_test = df_test[label]
    
    return X_train, X_test, y_train, y_test


def mostrar_matriz_confusion(y, y_pred):
    """
    Muestra la matriz de confusión para un modelo de clasificación binaria.
    """
    cm = confusion_matrix(y, y_pred)

    # Configurar la visualización
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Resto (0)", "Top Quintile (1)"]
    )

    # Graficar
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Matriz de Confusión (Datos Históricos)')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.grid(False) # Desactivar la grilla para que se vea más limpia
    plt.show()


def generar_ranking_predicciones(pipe, X_live, df, clase_objetivo=1, etiqueta_senal='Top_Quantile'):
    """
    Aplica el modelo predictivo a los datos recientes, calcula probabilidades, 
    genera señales de inversión y retorna los resultados ordenados.

    Parámetros:
    -----------
    pipe : sklearn.pipeline.Pipeline
        El pipeline entrenado con el preprocesador y el clasificador.
    X_live : pd.DataFrame
        El conjunto de datos de evaluación (mes actual en curso).
    df : pd.DataFrame
        El DataFrame original completo que contiene la columna 'Ticker'.
    clase_objetivo : int, por defecto 1
        La clase que representa el cuartil/quintil superior.
    etiqueta_senal : str, por defecto 'Top_Quantile'
        Etiqueta que se asignará a los casos positivos.

    Retorna:
    --------
    resultados_agrupados:
        DataFrame con Tickers, predicciones de clase, probabilidades y señales,
        ordenado de mayor a menor probabilidad.
    tickers_test:
        tickers presentes en los datos de evaluación.
    """
    
    # Predicciones de la clase y probabilidades
    y_pred_class = pipe.predict(X_live)
    y_pred_proba = pipe.predict_proba(X_live)

    # Extraer el modelo final del pipeline para conocer el orden de las clases
    rf_model = pipe.named_steps['model']

    # Identificar el índice de la probabilidad de la clase objetivo
    idx_top = list(rf_model.classes_).index(clase_objetivo)
    proba_top = y_pred_proba[:, idx_top]

    # Recuperar los Tickers correspondientes a X_live
    tickers_test = df.loc[X_live.index, 'Ticker']

    # Construir el DataFrame de resultados
    resultados_agrupados = pd.DataFrame({
        'Ticker': tickers_test.values,
        'Predicted_Class': y_pred_class,
        'Probability_Top': proba_top
    })

    # Generar la señal
    resultados_agrupados['Signal'] = np.where(
        resultados_agrupados['Predicted_Class'] == clase_objetivo, etiqueta_senal, 'Neutral'
    )

    # Ordenar resultados por la probabilidad de ser la clase objetivo
    resultados_agrupados = resultados_agrupados.sort_values(by='Probability_Top', ascending=False)
    
    return resultados_agrupados, tickers_test


def generar_reporte(df:pd.DataFrame, resultados_agrupados:pd.DataFrame)->pd.DataFrame:
    """
    Genera un reporte consolidado de predicciones y datos de empresas,
    exportándolo a un archivo CSV con la fecha actual.
    """
    # Se filtra df para mantener solo la fila más reciente de cada empresa
    df_latest = df.drop_duplicates(subset=['Ticker'], keep='last')

    df_reporte = resultados_agrupados.merge(df_latest, how='left', on='Ticker') 

    dia = datetime.now().day
    mes = datetime.now().month
    year = datetime.now().year

    # Crear carpeta si no existe y nombrar el archivo con la fecha
    reports_folder.mkdir(parents=True, exist_ok=True)
    nombre_archivo = f"{year}_{mes}_{dia}.csv"
    ruta_completa = reports_folder / nombre_archivo

    df_reporte.to_csv(ruta_completa, index=False)
    print(f'Reporte exportado en la carpeta de datos.')

    return df_reporte


def graficar_explicacion_shap(ticker_a_explicar, pipe, X_live, tickers, clase_objetivo=1, max_display=10):
    """
    Genera un gráfico de cascada (waterfall) con los valores SHAP para explicar
    la predicción de un modelo RandomForestClassifier dentro de un Pipeline.

    Parámetros:
    -----------
    ticker_a_explicar : str
        El ticker (símbolo financiero) que se desea evaluar (ej. 'XRX').
    pipe : sklearn.pipeline.Pipeline
        El pipeline entrenado que contiene los pasos 'pre' (ColumnTransformer) y 'model' (RandomForestClassifier).
    X_live : pd.DataFrame
        El conjunto de datos de predicción (features del mes actual en curso).
    tickers : pd.Series o np.ndarray
        Estructura con los Tickers correspondientes al índice de X_live.
    clase_objetivo : int, por defecto 1
        La clase del modelo para la cual se calculará la explicación (Top Quintile).
    max_display : int, por defecto 10
        Número máximo de características a mostrar en el gráfico de SHAP.
    """
    
    # Verificación temprana del ticker
    if ticker_a_explicar not in tickers.values:
        print(f"El ticker {ticker_a_explicar} no se encuentra en el conjunto de test.")
        return

    # Extraer los componentes del pipeline
    preprocessor = pipe.named_steps['pre']
    rf_model = pipe.named_steps['model']

    # Transformar los datos usando el preprocesador
    X_test_transformed = preprocessor.transform(X_live)
    feature_names = preprocessor.get_feature_names_out()

    # Crear un DataFrame con los datos transformados para que SHAP lea los nombres
    X_test_shap = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_live.index)

    # Obtener la posición del ticker y extraer su fila
    idx = np.where(tickers.values == ticker_a_explicar)[0][0]
    X_ticker_eval = X_test_shap.iloc[[idx]]

    # Inicializar el explicador y calcular los valores SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_ticker_eval)

    # Aislar la explicación específica para la clase objetivo
    idx_top = list(rf_model.classes_).index(clase_objetivo)
    shap_values_q5 = shap_values[..., idx_top]

    print(f"--- Explicación de la probabilidad de ser Quintil {clase_objetivo} para: {ticker_a_explicar} ---")

    # Crear la figura y visualizar
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_q5[0], max_display=max_display)
    plt.show()



# Funciones "legacy": ya no se utilizan en el modelado por clasificación

def obtener_metricas(y_real, y_pred, nombre_modelo):
    """
    Calcula métricas de evaluación para modelos de regresión.
    """
    mse = mean_squared_error(y_real, y_pred)
    
    
    return {
        'Modelo': nombre_modelo,
        'MAE': mean_absolute_error(y_real, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_real, y_pred)
    }


def procesar_resultados_prediccion(y_test, y_pred, tickers):
    """
    Consolida las predicciones, calcula residuos, agrupa por ticker y asigna un cluster de sesgo.
    
    Parámetros:
    - y_test (pd.Series): Valores observados/reales del conjunto de test.
    - y_pred (np.array o pd.Series): Valores predichos por el modelo.
    - tickers (pd.Series): Serie con los nombres de los tickers correspondientes al y_test.
    
    Retorna:
    - pd.DataFrame: DataFrame agrupado por Ticker con promedios y el Cluster asignado.
    """
    # Validación: asegurar que las longitudes coincidan
    if len(y_test) != len(y_pred):
        raise ValueError(f"Dimensión incorrecta: y_test ({len(y_test)}) y y_pred ({len(y_pred)}) no coinciden.")
    
    # Consolidar datos en el DataFrame de resultados
    resultados_df = pd.DataFrame({
        'Ticker': tickers,
        'Predicted': y_pred,
        'Observed': y_test
    })
    
    # Cálculo del residuo
    resultados_df['Residuals'] = resultados_df['Predicted'] - resultados_df['Observed']
    
    # 4. Agrupación (calcula el promedio por si hay >1 fecha por ticker y fija Ticker como índice)
    resultados_agrupados = resultados_df.groupby('Ticker')[['Predicted', 'Observed', 'Residuals']].mean()
    
    # 5. Generar el Cluster de forma vectorizada
    resultados_agrupados['Cluster'] = np.where(
        resultados_agrupados['Residuals'] >= 0, 
        'PositiveBias', 
        'NegativeBias'
    )
    
    return resultados_agrupados

def visualizar_resultados_predicciones(resultados_agrupados: pd.DataFrame):
    """
    Visualiza los resultados de predicciones vs valores reales, coloreando por cluster de sesgo.
    Además, imprime estadísticas de sesgo por cluster.
    """
    fig = px.scatter(
        resultados_agrupados, 
        x='Observed', 
        y='Predicted', 
        color='Cluster',
        # --- Mapeo de colores
        color_discrete_map={
            'PositiveBias': 'blue', 
            'NegativeBias': 'red'
        },

        hover_name=resultados_agrupados.index, 
        labels={
            'Observed': 'Valores Reales', 
            'Predicted': 'Predicciones', 
            'Cluster': 'Sesgo del Modelo'
        },
        title='Predicciones vs Reales (Agrupado por Ticker)'
    )

    # Línea de identidad perfecta
    min_val = resultados_agrupados['Observed'].min()
    max_val = resultados_agrupados['Observed'].max()
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color='black', dash='dash', width=2))
    
    fig.show()

    # Estadísticas por cluster a nivel Ticker
    positive_mask = resultados_agrupados['Cluster'] == 'PositiveBias'
    negative_mask = resultados_agrupados['Cluster'] == 'NegativeBias'

    print("\nEstadísticas por cluster (a nivel de Ticker):")
    print(f"Sesgos positivos: {positive_mask.sum()} tickers, residuo medio global: {resultados_agrupados.loc[positive_mask, 'Residuals'].mean():.4f}")
    print(f"Sesgos negativos: {negative_mask.sum()} tickers, residuo medio global: {resultados_agrupados.loc[negative_mask, 'Residuals'].mean():.4f}")