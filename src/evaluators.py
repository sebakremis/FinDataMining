# src/evaluators.py
"""
Módulo de funciones para la fase de Evaluación de Modelos (Modeling).
"""
import numpy as np
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

# Bloque principal para pruebas desde el terminal

def main():
    pass

if __name__ == "__main__":
    main()