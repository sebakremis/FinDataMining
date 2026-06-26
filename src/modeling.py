# src/modeling.py
"""
Módulo de la fase de modelado
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px


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


def split_ultimo(
    df: pd.DataFrame, 
    label: str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
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

def visualizar_resultados_predicciones(resultados_agrupados:pd.DataFrame):
    fig = px.scatter(
        resultados_agrupados, 
        x='Observed', 
        y='Predicted', 
        color='Cluster',
        hover_name=resultados_agrupados.index, 
        labels={'Observed':'Valores Reales', 'Predicted':'Predicciones', 'Cluster':'Sesgo del Modelo'},
        title='Predicciones vs Reales (Agrupado por Ticker)'
    )

    # Línea de identidad perfecta
    min_val = resultados_agrupados['Observed'].min()
    max_val = resultados_agrupados['Observed'].max()
    fig.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                line=dict(color='black', dash='dash', width=2))
    fig.show()

    # Estadísticas por cluster a nivel Ticker
    over_mask = resultados_agrupados['Cluster'] == 'PositiveBias'
    under_mask = resultados_agrupados['Cluster'] == 'NegativeBias'

    print("\nEstadísticas por cluster (a nivel de Ticker):")
    print(f"Overprediction: {over_mask.sum()} tickers, residuo medio global: {resultados_agrupados.loc[over_mask, 'Residuals'].mean():.4f}")
    print(f"Underprediction: {under_mask.sum()} tickers, residuo medio global: {resultados_agrupados.loc[under_mask, 'Residuals'].mean():.4f}")