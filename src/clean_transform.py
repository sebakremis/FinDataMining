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
    condicion_2 = clean_df['TotalRevenue_TTM'] < 0

    # Se asignan a NaN
    clean_df['TotalRevenue_TTM'] = np.where(condicion_2, np.nan, clean_df['TotalRevenue_TTM']) 

    # Caso 3:  Deuda negativa
    condicion_3_current_debt = (clean_df['CurrentDebt'] < 0)
    condicion_3_longterm_debt = (clean_df['LongTermDebt'] < 0)

    # Se asignan a NaN
    clean_df['CurrentDebt'] = np.where(condicion_3_current_debt, np.nan, clean_df['CurrentDebt']) 
    clean_df['LongTermDebt'] = np.where(condicion_3_longterm_debt, np.nan, clean_df['LongTermDebt'])

    # Caso 4:  Depreciación y Amortización negativa de yfinance
    condicion_4 = (clean_df['DepreciationAndAmortization_TTM'] < 0) & (clean_df['FinancialsSource']=='yfinance')

    # Se asignan a NaN
    clean_df.loc[condicion_4,'DepreciationAndAmortization_TTM'] = np.nan

    # Caso 5: Depreciación y Amortización negativa de simFin
    condicion_5 = (clean_df['DepreciationAndAmortization_TTM'] < 0) & (clean_df['FinancialsSource']=='simFin')

    # Se convierten a positivos
    clean_df.loc[condicion_5, 'DepreciationAndAmortization_TTM'] = clean_df.loc[condicion_5, 'DepreciationAndAmortization_TTM'].abs()

    return clean_df


def imputar_info(df:pd.DataFrame)->pd.DataFrame:
    """
    Para imputar manualmente si quedan missings remanentes en Sector e Industry
    """
    df_out = df.copy()
    casos = [
        # Se imputan con la tupla: (Ticker, Sector, Industry)
        ('MKSI', 'Technology', 'Scientific And Technical Instruments'),
        ('WTW', 'Financial Services', 'Insurance Brokers'),
        # Agregar nuevos casos aquí
        
    ]
    for ticker, sector, industry in casos:
        condicion = (df_out['Ticker'] == ticker)
        df_out.loc[condicion, 'Sector'] = sector
        df_out.loc[condicion, 'Industry'] = industry

    return df_out