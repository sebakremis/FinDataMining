"""
src/clean_transform.py
Módulo auxiliar de la fase de Transformación de Datos.
Trata las errores detectados en el dataset raw.
"""
import pandas as pd
import numpy as np


def corregir_anomalias(df:pd.DataFrame)->pd.DataFrame:
    """
    Efectua las correcciones de los casos analizados en el notebook de transformación.
    """   
    clean_df = df.copy()

    # Caso 1 - Error de importes en el Galance General
    condicion_1 = (clean_df['Ticker'] == 'IIIN') & (clean_df['Date'] == '2021-12-01')

    # Columnas del balance que tienen el error
    columnas_1 = [
        'CashAndCashEquivalents', 
        'TotalAssets', 
        'StockholdersEquity', 
        'CurrentAssets', 
        'CurrentLiabilities'
    ]

    # Se dividen por 1000
    clean_df.loc[condicion_1, columnas_1] /= 1000

    # Caso 2:  TotalRevenue negativo
    condicion_2 = clean_df['TotalRevenue'] < 0

    # Se asignan a NaN
    clean_df['TotalRevenue'] = np.where(condicion_2, np.nan, clean_df['TotalRevenue']) 

    # Caso 3:  Deuda negativa
    condicion_3 = (clean_df['CurrentDebt'] < 0) | (clean_df['LongTermDebt'] < 0)

    # Se eliminan los registros
    clean_df = clean_df[~condicion_3].reset_index(drop=True)

    # Caso 4:  Depreciación y Amortización negativa de yfinance
    condicion_4 = (df['DepreciationAndAmortization'] < 0) & (df['FinancialsSource']=='yfinance')

    # Se reemplazan por cero
    df.loc[condicion_4,'DepreciationAndAmortization'] = 0

    # Caso 5: Depreciación y Amortización negativa de simFin
    condicion_5 = (df['DepreciationAndAmortization'] < 0) & (df['FinancialsSource']=='simFin')

    # Se convierten a positivos
    df.loc[condicion_5, 'DepreciationAndAmortization'] = df.loc[condicion_5, 'DepreciationAndAmortization'].abs()

    return clean_df


def imputar_info(df:pd.DataFrame)->pd.DataFrame:
    """
    Se imputan manualmente los missings encontrados en Industry y Sector
    """
    df_out = df.copy()
    # Caso 1: MKSI
    condicion_1 = df_out['Ticker'] == 'MKSI'
    df_out.loc[condicion_1, 'Sector'] = 'Technology'
    df_out.loc[condicion_1, 'Industry'] = 'Scientific And Technical Instruments'

    return df_out