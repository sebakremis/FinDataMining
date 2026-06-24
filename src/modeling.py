# src/modeling.py
"""
Módulo de la fase de modelado
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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