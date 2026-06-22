"""
src/clean_transform.py
Módulo auxiliar de la fase de Transformación de Datos.
Trata las anomalías detectadas en el dataset raw.
"""
import pandas as pd
import numpy as np

def limpiar_data(df:pd.DataFrame)->pd.DataFrame:

    # Caso 1 - Error de importes en el Galance General: están multiplicados por 1.000
    clean_df = df.copy()
    condicion = (clean_df['Ticker'] == 'IIIN') & (clean_df['Date'] == '2021-12-01')

    # Columnas del balance que tienen el error
    columnas_balance = [
        'CashAndCashEquivalents', 
        'TotalAssets', 
        'StockholdersEquity', 
        'CurrentAssets', 
        'CurrentLiabilities'
    ]

    # Se dividen por 1000
    clean_df.loc[condicion, columnas_balance] /= 1000

    # Anomalias de signo
    # Caso 2: Asignar a NaN TotalRevenue negativo
    clean_df['TotalRevenue'] = np.where(clean_df['TotalRevenue'] < 0, np.nan, clean_df['TotalRevenue'])

    # Caso 3: Eliminar registros con deuda negativa
    condicion = (clean_df['CurrentDebt'] < 0) | (clean_df['LongTermDebt'] < 0)
    clean_df = clean_df[~condicion].reset_index(drop=True)

    # Caso 4: Reemplazar por cero caso con amortización negativa de yfinance
    clean_df.loc[condicion,'DepreciationAndAmortization'] = 0

    # Caso 5: Convertir a positivos los negativos que vienen de simFin
    clean_df['DepreciationAndAmortization'] = clean_df['DepreciationAndAmortization'].abs()

    return clean_df


