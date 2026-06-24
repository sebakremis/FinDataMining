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
    condicion_error_balance = (clean_df['Ticker'] == 'IIIN') & (clean_df['Date'] == '2021-12-01')

    # Columnas del balance que tienen el error
    columnas_error_balance = [
        'CashAndCashEquivalents', 
        'TotalAssets', 
        'StockholdersEquity', 
        'CurrentAssets', 
        'CurrentLiabilities'
    ]

    # Se dividen por 1000
    clean_df.loc[condicion_error_balance, columnas_error_balance] /= 1000

    # Anomalías de signo
    # Caso 2: Asignar a NaN TotalRevenue negativo
    condicion_revenue_negativo = clean_df['TotalRevenue'] < 0
    clean_df['TotalRevenue'] = np.where(condicion_revenue_negativo, np.nan, clean_df['TotalRevenue'])

    # Caso 3: Eliminar registros con deuda negativa
    condicion_deuda_negativa = (clean_df['CurrentDebt'] < 0) | (clean_df['LongTermDebt'] < 0)
    clean_df = clean_df[~condicion_deuda_negativa].reset_index(drop=True)

    # Caso 4: Reemplazar por cero caso con amortización negativa de yfinance
    condicion_depre_negativa_yfinance = (df['DepreciationAndAmortization'] < 0) & (df['FinancialsSource']=='yfinance')
    df.loc[condicion_depre_negativa_yfinance,'DepreciationAndAmortization'] = 0

    # Caso 5: Convertir a positivos los negativos que vienen de simFin
    condicion_depre_negativa_simfin = (df['DepreciationAndAmortization'] < 0) & (df['FinancialsSource']=='simFin')
    df.loc[condicion_depre_negativa_simfin, 'DepreciationAndAmortization'] = df.loc[condicion_depre_negativa_simfin, 'DepreciationAndAmortization'].abs()

    return clean_df


