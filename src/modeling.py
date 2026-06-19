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


def calcular_acceleration_features(df:pd.DataFrame, cols:list, reemplazar:bool= False)-> pd.DataFrame:
    for col in cols:
        try:
            # Se extrae el nombre de la variable
            feature_name = col.split('_')[0]

            # Calcular Acceleration: se define como la tasa de cambio a corto plazo menos la de largo.
            df[f'{feature_name}_Acceleration'] = df[f'{feature_name}_QoQ'] - df[f'{feature_name}_YoY']

            if reemplazar:
                # Se elimina la columna trimestral
                df = df.drop(f'{feature_name}_QoQ', axis=1)           

        except Exception as e:
            print(f"Error procesando columna {col}: {e}")
            continue

    return df