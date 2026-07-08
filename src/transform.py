"""
src/transform.py
Módulo de la fase de Transformación de Datos
"""

import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
from src.clean_transform import corregir_anomalias, imputar_info
from src.extract import extraer_info, obtener_cols_financieras
from src.config import (
    raw_data_file, clean_data_file, market_index_file
)

# --- Funciones de Limpieza de Datos ---


def limpiar_cadenas(texto):
    if not isinstance(texto, str): # Por si hay valores nulos (NaN)
        return texto
    
    # Se reemplaza el guión por espacio
    texto = texto.replace('-', ' ')

    # Dividir en palabras
    palabras = texto.split()

    # Procesar cada palabra: se cambia a 'And' si es '&', si no, se capitaliza
    palabras_procesadas = ['And' if p == '&' else p.capitalize() for p in palabras]
    # Unimos todo sin espacios
    return ' '.join(palabras_procesadas)


def limpiar_industry_y_sector(df:pd.DataFrame)->pd.DataFrame:
    """
    Aplica la limpieza de cadenas y convierte a variables de tipo category.
    
    """
    df_out = df.copy()
    df_out['Industry'] = df_out['Industry'].apply(limpiar_cadenas)
    df_out['Sector'] = df_out['Sector'].apply(limpiar_cadenas)

    # Convertir Sector y Industry a tipo category
    df_out['Sector'] = df_out['Sector'].astype('category')
    df_out['Industry'] = df_out['Industry'].astype('category')

    return df_out   


def recuperar_info(df:pd.DataFrame)->pd.DataFrame:
    df_out = df.copy()
    tickers = df_out[df_out['Sector'].isna() | df_out['Industry'].isna()]['Ticker'].unique().tolist()
    if len(tickers)==0:
        print("No se encontraron valores perdidos de Sectores/Industrias.")
        return df_out

    info_recuperada = extraer_info(tickers)

    # Validar que se hayan recuperado los datos
    if info_recuperada.empty or 'Sector' not in info_recuperada.columns:
        print("No se recuperó información válida de Sectores/Industrias.")
        return df_out

    for ticker in tickers:
        # Se filtra el dataframe de info recuperada para el ticker actual
        info_ticker = (info_recuperada[info_recuperada['Ticker'] == ticker])

        if not info_ticker.empty:
            sector_val = info_ticker['Sector'].values[0]
            industry_val = info_ticker['Industry'].values[0]

            df_out.loc[df_out['Ticker']==ticker, 'Sector'] = sector_val
            df_out.loc[df_out['Ticker']==ticker, 'Industry'] = industry_val
    
    return df_out


def columnas_en_millones(df:pd.DataFrame)->pd.DataFrame:
    cols = obtener_cols_financieras(incluirTTM=True)
    cols.append('Volume') # se convierte también el volumen
    cols.append('CashAndCashEquivalents') # no está en la lista de columnas de yfinance
    set_cols = set(cols)
    df[list(set_cols)] = df[list(set_cols)] / 10**6
    return df


# --- Tratamiento de Missings

def mostrar_missings(df:pd.DataFrame)->pd.Series:
    """
    Calcula la proporción de valores nulos por columna usando operaciones vectorizadas.
    """
    # Se calcula la proporción de nulos para todas las columnas
    missings = df.isna().mean()
    
    # Se filtran donde el valor es mayor a 0, ordenando luego de mayor a menor
    return missings[missings > 0].sort_values(ascending=False)


import pandas as pd

def imputar_equivalencias_financieras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores nulos en columnas financieras utilizando identidades contables.
    """
    # Trabajar sobre una copia para no alterar el dataframe original accidentalmente
    df_imputado = df.copy()

    # =========================================================================
    # 1. IMPUTACIONES DE ESTADO DE RESULTADOS (P&L)
    # =========================================================================
    
    # Se imputan a cero los valores faltantes en "DepreciationAndAmortization".
    # Se asume que es cero ya que empresas donde no es relevante no lo informan.
    df_imputado['DepreciationAndAmortization_TTM'] = df_imputado['DepreciationAndAmortization_TTM'].fillna(0)

    # Se imputa 'EBITDA' mediante la ecuación: EBITDA = Operating Income + D&A
    cond_falta_ebitda = df_imputado['EBITDA_TTM'].isna() & df_imputado['OperatingIncome_TTM'].notna()
    df_imputado.loc[cond_falta_ebitda, 'EBITDA_TTM'] = (
        df_imputado.loc[cond_falta_ebitda, 'OperatingIncome_TTM'] + 
        df_imputado.loc[cond_falta_ebitda, 'DepreciationAndAmortization_TTM']
    )

    # =========================================================================
    # 2. IMPUTACIONES DE FLUJO DE CAJA (CASH FLOW)
    # =========================================================================
    
    # Imputar FreeCashFlow (Fórmula: FreeCashFlow = OperatingCashFlow - Capital Expenditure)
    # Se usa .abs() en CapEx para estandarizar salidas de caja y restarlas correctamente
    if 'FreeCashFlow_TTM' in df_imputado.columns and 'OperatingCashFlow_TTM' in df_imputado.columns and 'CapitalExpenditure_TTM' in df_imputado.columns:
        cond_falta_fcf = df_imputado['FreeCashFlow_TTM'].isna() & df_imputado['OperatingCashFlow_TTM'].notna() & df_imputado['CapitalExpenditure_TTM'].notna()
        df_imputado.loc[cond_falta_fcf, 'FreeCashFlow_TTM'] = (
            df_imputado.loc[cond_falta_fcf, 'OperatingCashFlow_TTM'] - 
            df_imputado.loc[cond_falta_fcf, 'CapitalExpenditure_TTM'].abs()
        )

    # =========================================================================
    # 3. IMPUTACIONES DE BALANCE GENERAL (DEUDA)
    # =========================================================================
    
    columnas_deuda = ['TotalDebt', 'CurrentDebt', 'LongTermDebt']
    if not all(col in df_imputado.columns for col in columnas_deuda):
        print("Advertencia: Faltan columnas de deuda. Se omite imputación de deuda.")
    else:
        # CASO A: Rescate de TotalDebt (TotalDebt = CurrentDebt + LongTermDebt)
        cond_falta_total_debt = df_imputado['TotalDebt'].isna() & df_imputado['CurrentDebt'].notna() & df_imputado['LongTermDebt'].notna()
        df_imputado.loc[cond_falta_total_debt, 'TotalDebt'] = (
            df_imputado.loc[cond_falta_total_debt, 'CurrentDebt'] + 
            df_imputado.loc[cond_falta_total_debt, 'LongTermDebt']
        )

        # CASO B: Si TotalDebt es 0, las componentes son 0
        condicion_cero = (df_imputado['TotalDebt'] == 0).fillna(False)
        df_imputado.loc[condicion_cero & df_imputado['CurrentDebt'].isna(), 'CurrentDebt'] = 0
        df_imputado.loc[condicion_cero & df_imputado['LongTermDebt'].isna(), 'LongTermDebt'] = 0

        # CASO C: Deducción por resta (CurrentDebt = TotalDebt - LongTermDebt)
        cond_falta_current = df_imputado['CurrentDebt'].isna() & df_imputado['TotalDebt'].notna() & df_imputado['LongTermDebt'].notna()
        df_imputado.loc[cond_falta_current, 'CurrentDebt'] = (
            df_imputado.loc[cond_falta_current, 'TotalDebt'] - 
            df_imputado.loc[cond_falta_current, 'LongTermDebt']
        )

        cond_falta_long = df_imputado['LongTermDebt'].isna() & df_imputado['TotalDebt'].notna() & df_imputado['CurrentDebt'].notna()
        df_imputado.loc[cond_falta_long, 'LongTermDebt'] = (
            df_imputado.loc[cond_falta_long, 'TotalDebt'] - 
            df_imputado.loc[cond_falta_long, 'CurrentDebt']
        )

        # CASO D: Heurística 80/20
        RATIO_LTD = 0.8
        RATIO_CD = 1 - RATIO_LTD  # 0.2

        cond_solo_total = df_imputado['TotalDebt'].notna() & df_imputado['CurrentDebt'].isna() & df_imputado['LongTermDebt'].isna()
        df_imputado.loc[cond_solo_total, 'LongTermDebt'] = df_imputado.loc[cond_solo_total, 'TotalDebt'] * RATIO_LTD
        df_imputado.loc[cond_solo_total, 'CurrentDebt']  = df_imputado.loc[cond_solo_total, 'TotalDebt'] * RATIO_CD

        cond_solo_long = df_imputado['LongTermDebt'].notna() & df_imputado['TotalDebt'].isna() & df_imputado['CurrentDebt'].isna()
        df_imputado.loc[cond_solo_long, 'TotalDebt']   = df_imputado.loc[cond_solo_long, 'LongTermDebt'] / RATIO_LTD
        df_imputado.loc[cond_solo_long, 'CurrentDebt'] = df_imputado.loc[cond_solo_long, 'TotalDebt'] * RATIO_CD

        cond_solo_current = df_imputado['CurrentDebt'].notna() & df_imputado['TotalDebt'].isna() & df_imputado['LongTermDebt'].isna()
        df_imputado.loc[cond_solo_current, 'TotalDebt']    = df_imputado.loc[cond_solo_current, 'CurrentDebt'] / RATIO_CD
        df_imputado.loc[cond_solo_current, 'LongTermDebt'] = df_imputado.loc[cond_solo_current, 'TotalDebt'] * RATIO_LTD

        df_imputado['CurrentDebt'] = df_imputado['CurrentDebt'].clip(lower=0)
        df_imputado['LongTermDebt'] = df_imputado['LongTermDebt'].clip(lower=0)

    # =========================================================================
    # 4. IMPUTACIONES DE BALANCE GENERAL (PASIVOS)
    # =========================================================================
    
    columnas_pasivo = ['TotalLiabilities', 'CurrentLiabilities', 'TotalNoncurrentLiabilities']
    if not all(col in df_imputado.columns for col in columnas_pasivo):
        print("Advertencia: Faltan columnas de pasivos. Se omite imputación de pasivos.")
    else:
        # CASO A: Rescate de TotalLiabilities
        cond_falta_total_liab = df_imputado['TotalLiabilities'].isna() & df_imputado['CurrentLiabilities'].notna() & df_imputado['TotalNoncurrentLiabilities'].notna()
        df_imputado.loc[cond_falta_total_liab, 'TotalLiabilities'] = (
            df_imputado.loc[cond_falta_total_liab, 'CurrentLiabilities'] + 
            df_imputado.loc[cond_falta_total_liab, 'TotalNoncurrentLiabilities']
        )

        # CASO B: Si el Pasivo Total es 0, sus componentes son 0
        condicion_cero_liab = (df_imputado['TotalLiabilities'] == 0).fillna(False)
        df_imputado.loc[condicion_cero_liab & df_imputado['CurrentLiabilities'].isna(), 'CurrentLiabilities'] = 0
        df_imputado.loc[condicion_cero_liab & df_imputado['TotalNoncurrentLiabilities'].isna(), 'TotalNoncurrentLiabilities'] = 0

        # CASO C: Deducción por resta
        # C1) Falta Corto Plazo (Current)
        cond_falta_current_liab = df_imputado['CurrentLiabilities'].isna() & df_imputado['TotalLiabilities'].notna() & df_imputado['TotalNoncurrentLiabilities'].notna()
        df_imputado.loc[cond_falta_current_liab, 'CurrentLiabilities'] = (
            df_imputado.loc[cond_falta_current_liab, 'TotalLiabilities'] - 
            df_imputado.loc[cond_falta_current_liab, 'TotalNoncurrentLiabilities']
        )

        # C2) Falta Largo Plazo (NonCurrent)
        cond_falta_long_liab = df_imputado['TotalNoncurrentLiabilities'].isna() & df_imputado['TotalLiabilities'].notna() & df_imputado['CurrentLiabilities'].notna()
        df_imputado.loc[cond_falta_long_liab, 'TotalNoncurrentLiabilities'] = (
            df_imputado.loc[cond_falta_long_liab, 'TotalLiabilities'] - 
            df_imputado.loc[cond_falta_long_liab, 'CurrentLiabilities']
        )

        # CASO D: Heurística 80/20 cuando SOLO SE TIENE 1 DE LAS 3 COLUMNAS
        RATIO_NCL = 0.8  # Non-Current Liabilities (Largo Plazo)
        RATIO_CL = 1 - RATIO_NCL  # Current Liabilities (Corto Plazo)

        # D1) SOLO se tiene Total (Faltan Current y NonCurrent)
        cond_solo_total_liab = df_imputado['TotalLiabilities'].notna() & df_imputado['CurrentLiabilities'].isna() & df_imputado['TotalNoncurrentLiabilities'].isna()
        df_imputado.loc[cond_solo_total_liab, 'TotalNoncurrentLiabilities'] = df_imputado.loc[cond_solo_total_liab, 'TotalLiabilities'] * RATIO_NCL
        df_imputado.loc[cond_solo_total_liab, 'CurrentLiabilities']  = df_imputado.loc[cond_solo_total_liab, 'TotalLiabilities'] * RATIO_CL

        # D2) SOLO se tiene NonCurrent (Faltan Total y Current)
        cond_solo_long_liab = df_imputado['TotalNoncurrentLiabilities'].notna() & df_imputado['TotalLiabilities'].isna() & df_imputado['CurrentLiabilities'].isna()
        df_imputado.loc[cond_solo_long_liab, 'TotalLiabilities']   = df_imputado.loc[cond_solo_long_liab, 'TotalNoncurrentLiabilities'] / RATIO_NCL
        df_imputado.loc[cond_solo_long_liab, 'CurrentLiabilities'] = df_imputado.loc[cond_solo_long_liab, 'TotalLiabilities'] * RATIO_CL

        # D3) SOLO se tiene Current (Faltan Total y NonCurrent)
        cond_solo_current_liab = df_imputado['CurrentLiabilities'].notna() & df_imputado['TotalLiabilities'].isna() & df_imputado['TotalNoncurrentLiabilities'].isna()
        df_imputado.loc[cond_solo_current_liab, 'TotalLiabilities']    = df_imputado.loc[cond_solo_current_liab, 'CurrentLiabilities'] / RATIO_CL
        df_imputado.loc[cond_solo_current_liab, 'TotalNoncurrentLiabilities'] = df_imputado.loc[cond_solo_current_liab, 'TotalLiabilities'] * RATIO_NCL

    # =========================================================================
    # 5. IMPUTACIONES DE BALANCE GENERAL (ACTIVOS CORRIENTES)
    # =========================================================================
    
    # Verificar disponibilidad de las columnas ancla básicas
    if 'CurrentAssets' in df_imputado.columns and 'TotalAssets' in df_imputado.columns and 'Sector' in df_imputado.columns:
        
        # Paso A: Calcular el ratio temporal observado de Activo Corriente sobre Activo Total
        df_imputado['_ratio_ca_ta'] = df_imputado['CurrentAssets'] / df_imputado['TotalAssets']
        
        # Paso B: Obtener la mediana de este ratio por Sector
        # (Se usa transform para que mantenga la misma longitud del DataFrame)
        mediana_sector_ratio = df_imputado.groupby('Sector', observed=True)['_ratio_ca_ta'].transform('median')
        
        # Respaldo: Si un sector es completamente nuevo o tiene puros nulos, usar la mediana global
        mediana_global_ratio = df_imputado['_ratio_ca_ta'].median()
        mediana_sector_ratio = mediana_sector_ratio.fillna(mediana_global_ratio)
        
        # Paso C: Imputar estimación basada en el tamaño de la empresa (TotalAssets)
        cond_falta_ca = df_imputado['CurrentAssets'].isna() & df_imputado['TotalAssets'].notna()
        df_imputado.loc[cond_falta_ca, 'CurrentAssets'] = (
            df_imputado.loc[cond_falta_ca, 'TotalAssets'] * mediana_sector_ratio
        )
        
        # Eliminar columna auxiliar de cálculo
        df_imputado.drop(columns=['_ratio_ca_ta'], errors='ignore', inplace=True)

    # Paso D: Validar contra el Piso Contable Estricto (CashAndCashEquivalents)
    if 'CurrentAssets' in df_imputado.columns and 'CashAndCashEquivalents' in df_imputado.columns:
        # D1) Si aún queda algún missing aislado en CurrentAssets, lo igualamos al menos a su efectivo conocido
        cond_todavia_missing = df_imputado['CurrentAssets'].isna() & df_imputado['CashAndCashEquivalents'].notna()
        df_imputado.loc[cond_todavia_missing, 'CurrentAssets'] = df_imputado.loc[cond_todavia_missing, 'CashAndCashEquivalents']
        
        # D2) Control de consistencia: El activo corriente total no puede ser inferior a la caja reportada
        cond_ca_menor_que_cash = df_imputado['CurrentAssets'] < df_imputado['CashAndCashEquivalents']
        df_imputado.loc[cond_ca_menor_que_cash, 'CurrentAssets'] = df_imputado.loc[cond_ca_menor_que_cash, 'CashAndCashEquivalents']


        # SEGURIDAD: Evitar pasivos negativos por discrepancias en reportes financieros
        df_imputado['CurrentLiabilities'] = df_imputado['CurrentLiabilities'].clip(lower=0)
        df_imputado['TotalNoncurrentLiabilities'] = df_imputado['TotalNoncurrentLiabilities'].clip(lower=0)

    return df_imputado


def imputar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por 'Ticker' y aplica una media o mediana móvil a las columnas numéricas 
    de un DataFrame para imputar valores nulos, dependiendo de su asimetría.
    """
    # Creamos una copia para no modificar el DataFrame original
    df_resultado = df.copy()
    umbral = 3 
    
    # Se seleccionan las columnas numéricas
    cols_numericas = df_resultado.select_dtypes(include=[np.number]).columns
    
    # Función auxiliar que se aplicará a cada Ticker
    def procesar_grupo(serie):
        sesgo = serie.skew()
                    
        # Se separa según el valor absoluto del umbral.
        # pd.notna(sesgo) evita un warning de Pandas al intentar comparar NaN < 3
        if pd.notna(sesgo) and abs(sesgo) < umbral:
            # Simétricas: Media móvil (mean)
            rolling_vals = serie.rolling(window=4, min_periods=1).mean()
        else:
            # Asimétricas (o NaN por falta de datos): Mediana móvil (median)
            rolling_vals = serie.rolling(window=4, min_periods=1).median()
            
        return serie.fillna(rolling_vals)

    # Iteramos sobre las numéricas y aplicamos la función agrupando por Ticker
    for col in cols_numericas:
        df_resultado[col] = df_resultado.groupby('Ticker')[col].transform(procesar_grupo)
        
    return df_resultado


def quitar_nulos_relevantes(df:pd.DataFrame, cols_no_relevantes:list[str]=[])->pd.DataFrame:
    df_out = df.copy()
    cols_numericas = df_out.select_dtypes(include="number").columns
    cols_relevantes = [col for col in cols_numericas if col not in cols_no_relevantes]
    df_out = df_out.dropna(subset=cols_relevantes)

    return df_out


def imputar_crecimientos(df:pd.DataFrame, cols:list[str])->pd.DataFrame:
    """
    Imputa las columnas de crecimiento con un valor neutral,
    igual a la tasa de inflación promedio de EE.UU.
    """
    df_out = df.copy()

    # Constantes macroeconómicas neutrales
    YOY_NEUTRAL = 0.0300
    QOQ_NEUTRAL = 0.0074

    # Se regeneran los nombres de las columnas de crecimiento QoQ y YoY
    cols_QoQ = [f'{col}_QoQ' for col in cols]
    cols_YoY = [f'{col}_YoY' for col in cols]  
    crecimiento_cols = cols_QoQ + cols_YoY

    for col in crecimiento_cols:
        if 'YoY' in col:
            # Crear la variable flag
            df_out[f'{col}_IsMissing'] = df_out[col].isna().astype(int)
            
            # Imputar con tasa neutral anual
            df_out[col] = df_out[col].fillna(YOY_NEUTRAL)

        elif 'QoQ' in col:   
            # Crear la variable flag
            df_out[f'{col}_IsMissing'] = df_out[col].isna().astype(int)
            
            # Imputar con tasa neutral trimestral
            df_out[col] = df_out[col].fillna(QOQ_NEUTRAL)

    return df_out


# --- Funciones de Feature Engineering ---

def crear_years_since_added(df:pd.DataFrame)->pd.DataFrame:
    #  Pasar DateAdded a formato datetime, los NaN se vuelven NaT (not a time)
    df['DateAdded'] = pd.to_datetime(df['DateAdded'], errors='coerce')

    # Convertir a YearsSinceAdded, aqui los nulos regresan a NaN
    df['YearsSinceAdded'] = round(((pd.Timestamp.now() - df['DateAdded']).dt.days / 365.25), 0)

    # Se asignan a cero años los tickers que no pertenecen al Índice S&P 500
    df['YearsSinceAdded'] = df['YearsSinceAdded'].fillna(0)

    # Eliminar la columna original
    df.drop('DateAdded', axis=1, inplace=True)

    return df


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame limpio de precios mensuales y datos fundamentales alineados, 
    y calcula métricas financieras históricas.
    """
    # Definir columnas necesarias para calcular
    cols_necesarias = [
        'Ticker', 
        'Date', 
        'Open', 
        'BasicAverageShares_TTM', 
        'TotalDebt', 
        'CashAndCashEquivalents', 
        'NetIncome_TTM', 
        'EBITDA_TTM', 
        'StockholdersEquity', 
        'OperatingIncome_TTM', 
        'TotalRevenue_TTM', 
        'TotalAssets', 
        'CurrentAssets', 
        'CurrentLiabilities', 
        'FreeCashFlow_TTM', 
        'CapitalExpenditure_TTM'
    ]

    for col in cols_necesarias:
        if col not in df.columns:
            print(f"Advertencia: Falta '{col}'. No se calcularán métricas.")
            return df
    
    # Se trabaja sobre una copia
    df_metrics = df.copy()

    # Asegurar ordenamiento por fecha   
    df_metrics = df_metrics.sort_values(by=['Ticker', 'Date'])

    # Calcular Capitalización Bursátil expresada en millones
    df_metrics['MarketCap'] = (df_metrics['Open'] * df_metrics['BasicAverageShares_TTM'])

    # Preparar deuda y efectivo para el EnterpriseValue = MarketCap + Deuda Total - Efectivo
    # (Nota: Asume que 'CurrentDebt' y 'LongTermDebt' existen si 'TotalDebt' es nulo)
    if 'CurrentDebt' in df_metrics.columns and 'LongTermDebt' in df_metrics.columns:
        deuda_total = df_metrics['TotalDebt'].fillna(
            df_metrics['CurrentDebt'].fillna(0) + df_metrics['LongTermDebt'].fillna(0)
        )
    else:
        deuda_total = df_metrics['TotalDebt'].fillna(0)
        
    efectivo = df_metrics['CashAndCashEquivalents'].fillna(0)
    df_metrics['EnterpriseValue'] = df_metrics['MarketCap'] + deuda_total - efectivo

    # --- RATIOS DE VALUACIÓN (Optimizados para ML: Yields) ---
    df_metrics['EarningsYield'] = df_metrics['NetIncome_TTM'] / df_metrics['MarketCap']
    df_metrics['EbitdaYield'] = df_metrics['EBITDA_TTM'] / df_metrics['EnterpriseValue']
    df_metrics['RevenueYield'] = df_metrics['TotalRevenue_TTM'] / df_metrics['EnterpriseValue']
    df_metrics['BookToMarket'] = df_metrics['StockholdersEquity'] / df_metrics['MarketCap']
    df_metrics['AssetToMarket'] = df_metrics['TotalAssets'] / df_metrics['MarketCap']

    # --- RATIOS DE RENTABILIDAD Y MÁRGENES ---
    df_metrics['OperatingMargins'] = df_metrics['OperatingIncome_TTM'] / df_metrics['TotalRevenue_TTM']
    df_metrics['ProfitMargins'] = df_metrics['NetIncome_TTM'] / df_metrics['TotalRevenue_TTM']
    df_metrics['ReturnOnAssets'] = df_metrics['NetIncome_TTM'] / df_metrics['TotalAssets']
    
    # Manejo del ROE (Corrección de signo si Patrimonio <= 0)
    df_metrics['ReturnOnEquity'] = np.where(
        df_metrics['StockholdersEquity'] <= 0,
        -np.abs(df_metrics['NetIncome_TTM'] / df_metrics['StockholdersEquity']),
        df_metrics['NetIncome_TTM'] / df_metrics['StockholdersEquity']
    )

    # --- RATIOS DE LIQUIDEZ Y SOLVENCIA ---
    df_metrics['DebtToEquity'] = deuda_total / df_metrics['StockholdersEquity']
    df_metrics['CurrentRatio'] = df_metrics['CurrentAssets'] / df_metrics['CurrentLiabilities']
        
    # --- OTRAS MÉTRICAS ---
    # Apalancamiento 
    df_metrics['NetDebtToEbitda'] = (deuda_total - df_metrics['CashAndCashEquivalents']) / df_metrics['EBITDA_TTM']

    # Free Cash Flow Conversion
    df_metrics['FcfToEbitda'] = df_metrics['FreeCashFlow_TTM'] / df_metrics['EBITDA_TTM']

    # Capital Intensity
    df_metrics['CapExToRevenue'] = np.abs(df_metrics['CapitalExpenditure_TTM']) / df_metrics['TotalRevenue_TTM']
    
    # Reemplazar por NaN todos los infinitos generados por divisiones por cero
    df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Manejo de valores inválidos: NaN en DebtToEquity si el patrimonio es negativo:
    mask_patrimonio_invalido = df_metrics['StockholdersEquity'] <= 0
    df_metrics.loc[mask_patrimonio_invalido, 'DebtToEquity'] = np.nan
  
    # --- LIMPIEZA FINAL: REDONDEAR COLUMNAS ---
    cols_a_redondear = [
        'EarningsYield', 'BookToMarket', 'EbitdaYield', 'OperatingMargins', 
        'ProfitMargins', 'ReturnOnEquity', 'ReturnOnAssets', 'DebtToEquity', 'CurrentRatio'
    ]
    
    df_metrics[cols_a_redondear] = df_metrics[cols_a_redondear].round(6)

    return df_metrics


def convertir_volumen_a_adv(df:pd.DataFrame)->pd.DataFrame:
    df['AverageDailyVolume'] = df['Volume'] / 21 # estimado de 21 días hábiles por mes
    df.drop('Volume', axis=1, inplace=True)
    return df


def calcular_crecimientos(df:pd.DataFrame, crecimiento_cols:list[str])->pd.DataFrame:
    """
    - Calcula crecimiento interanual (Year-over-Year, YoY) - Ventana de 4 trimestres 
    - Calcula crecimiento Trimestral (QoQ) - Ventana de 1 trimestre
    - Se aplica .abs() en el denominador para corregir matemáticamente la dirección del 
    crecimiento si el período anterior era negativo.
    """
    df_out = df.copy()

    for col in crecimiento_cols:
        prev_rev_12 = df_out.groupby('Ticker')[col].shift(12)
        df_out[f'{col}_YoY'] = (df_out[col] - prev_rev_12) / prev_rev_12.abs()
        prev_rev_3 = df_out.groupby('Ticker')[col].shift(3)
        df_out[f'{col}_QoQ'] = (df_out[col] - prev_rev_3) / prev_rev_3.abs()

        # Acotar entre -1000% y +1000% si quedan resultados absurdos al dividir entre cero o números pequeños
        df_out[f'{col}_YoY'] = df_out[f'{col}_YoY'].clip(lower=-10.0, upper=10.0)
        df_out[f'{col}_QoQ'] = df_out[f'{col}_QoQ'].clip(lower=-10.0, upper=10.0)

    return df_out


def calcular_aceleraciones(df:pd.DataFrame, cols:list)-> pd.DataFrame:
    df_out = df.copy()
    for col in cols:
        try:
            # Se extrae el nombre de la variable
            #feature_name = col.split('_')[0]

            # Calcular Acceleration: se define como la tasa de cambio a corto plazo menos la de largo.
            df_out[f'{col}_Acceleration'] = df[f'{col}_QoQ'] - df[f'{col}_YoY']        

        except Exception as e:
            print(f"Error procesando columna {col}: {e}")
            continue

    return df_out


def calcular_lag(df:pd.DataFrame, cols:list,months:int=1)->pd.DataFrame:
    for col in cols:
        df[col+f'_Lag{months}'] = df[col].shift(months)

    df.drop(columns=cols, inplace=True)
    return df


import pandas as pd

import pandas as pd

def calcular_retornos_y_betas(df: pd.DataFrame, df_index: pd.DataFrame, ventana: int = 12, min_periodos: int = 6) -> pd.DataFrame:
    """
    Calcula retornos (basados en Open), ShortTermBeta y Variance (Features).
    No es necesario aplicar Lag a las variables calculadas para evitar Lookahead Bias, 
    ya que el precio Open es conocido al obtenerlo.
    Calcula además el MonthlyExcessReturn (Label).
    Incluye un flag 'Return_IsMissing' para registrar imputaciones.
    """        
    ticker_mercado = df_index['Ticker'].iloc[0]
    
    # Preparar datos uniendo panel e índice
    df_unido = pd.concat([df, df_index], ignore_index=True)
    
    # Pivotar para aislar Open y Close
    df_open = df_unido.pivot(index='Date', columns='Ticker', values='Open').sort_index()
    df_close = df_unido.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    
    # ==========================================
    # BLOQUE 1: FEATURES (Usa Open a Open)
    # ==========================================

    # Calcular retornos en bruto, extraer flag y luego imputar
    df_retornos_raw = df_open.pct_change(fill_method=None)
    df_return_is_missing = df_retornos_raw.isna().astype(int)
    df_retornos_open = df_retornos_raw.fillna(0)
    
    retornos_mercado_open = df_retornos_open[ticker_mercado]
    df_activos_open = df_retornos_open.drop(columns=[ticker_mercado])
    df_missing_activos = df_return_is_missing.drop(columns=[ticker_mercado])
    
    # Calcular estadísticas móviles para las features
    varianza_mercado = retornos_mercado_open.rolling(window=ventana, min_periods=min_periodos).var()
    covarianzas = df_activos_open.rolling(window=ventana, min_periods=min_periodos).cov(retornos_mercado_open)
    df_betas = covarianzas.div(varianza_mercado, axis=0)
    df_varianzas = df_activos_open.rolling(window=ventana, min_periods=min_periodos).var()
      
    # =========================================================
    # BLOQUE 2: TARGET/LABEL BASE (Usa Close vs Open intra-mes)
    # =========================================================

    # Retorno intra-mes: (Close / Open) - 1
    df_retornos_intra = (df_close / df_open) - 1
    
    retornos_mercado_intra = df_retornos_intra[ticker_mercado]
    df_activos_intra = df_retornos_intra.drop(columns=[ticker_mercado])
    
    # Exceso de retorno intra-mes (Activo - Mercado)
    df_exceso = df_activos_intra.sub(retornos_mercado_intra, axis=0)
    
    # ==========================================
    # BLOQUE 3: CONSOLIDACIÓN (Melt y Merge)
    # ==========================================

    # Transformar features a formato largo
    df_ret_long = df_activos_open.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='MonthlyReturn'
    )
    df_beta_long = df_betas.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='ShortTermBeta'
    )
    df_var_long = df_varianzas.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='Variance'
    )
    df_missing_long = df_missing_activos.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='Return_IsMissing'
    )
    
    # Transformar target base a formato largo
    df_exceso_long = df_exceso.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='MonthlyExcessReturn'
    )
    
    # Consolidar todo en un solo DataFrame temporal
    df_features_and_target = (df_ret_long
                       .merge(df_beta_long, on=['Date', 'Ticker'])
                       .merge(df_var_long, on=['Date', 'Ticker'])
                       .merge(df_missing_long, on=['Date', 'Ticker'])
                       .merge(df_exceso_long, on=['Date', 'Ticker']))
    
    # Unir con el DataFrame original
    df_final = pd.merge(df, df_features_and_target, on=['Date', 'Ticker'], how='left')
    
    return df_final


def categorizar_en_cuantiles(df: pd.DataFrame, columna: str, num_cuantiles: int = 5) -> pd.DataFrame:
    """
    Transforma una columna continua en cuantiles cross-sectional (agrupando por 'Date').
    Convierte el resultado a tipo 'category' de Pandas y elimina la columna continua original.
    
    Parámetros:
    - df: DataFrame con los datos en formato panel (debe contener la columna 'Date').
    - columna: Nombre de la columna continua a transformar.
    - num_cuantiles: Cantidad de cuantiles (por defecto 5 para quintiles).
    
    Retorna:
    - Un nuevo DataFrame con la columna categórica añadida y la original eliminada.
    """
    # Trabajar sobre una copia para evitar el warning 'SettingWithCopy' o modificar el df original
    df_out = df.copy()
    
    # Definir el nombre de la nueva columna dinámicamente
    nueva_columna = f"{columna}_Quantile"
    
    # Función interna para manejar NaNs y fechas con pocos datos de forma segura
    def safe_qcut(group):
        valid_data = group.dropna()
        # Si no hay suficientes activos en ese mes para armar los cuantiles, devolver NaN
        if len(valid_data) < num_cuantiles:
            return pd.Series(np.nan, index=group.index)
        try:
            # labels=False devuelve de 0 a N-1. Sumamos 1 para que el rango sea de 1 a N.
            quintiles = pd.qcut(valid_data, q=num_cuantiles, labels=False, duplicates='drop') + 1
            return quintiles.reindex(group.index)
        except ValueError:
            return pd.Series(np.nan, index=group.index)
            
    # Aplicar la categorización cross-sectional (mes a mes)
    df_out[nueva_columna] = (
        df_out.groupby('Date')[columna]
        .transform(safe_qcut)
    )
    
    # Convertir explícitamente el tipo de dato a 'category'
    df_out[nueva_columna] = df_out[nueva_columna].astype('category')
    
    # Eliminar la columna original
    df_out = df_out.drop(columns=[columna])
    
    return df_out


# --- Funciones de Análisis Exploratorio de Datos ---

def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()


def graficar_cat(col):
     """
     Gráfico de barras para variables categóricas.
     """
     if col.dtypes == 'category':
        fig = px.bar(col.value_counts())
        #fig = sns.countplot(x=col)
        return(fig)


def graficar(col):
     """
     Función general para aplicar al archivo por columnas, detectando el tipo de variable y aplicando el gráfico adecuado.
     """
     if col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        graficar_cat(col).show()


def mostrar_asimetrias(df:pd.DataFrame):
    print(df.select_dtypes(include="number").skew().sort_values(ascending=False).to_string())


# --- Funciones de Transformación

def transformar_yeo_johnson(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df_out = df.copy()

    # Inicializar el PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True)

    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}_Yeo'] = pt.fit_transform(df_out[[col]])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out


def transformar_log(df: pd.DataFrame, cols: list, calculo_1p: bool = False) -> pd.DataFrame:
    # Creamos una copia para no alterar el DataFrame original accidentalmente
    df_out = df.copy()
    
    # Se define la función y el sufijo
    if calculo_1p:
        funcion_log = np.log1p
        sufijo = '_Log1p'
    else:
        funcion_log = np.log
        sufijo = '_Log'
        
    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}{sufijo}'] = funcion_log(df_out[col])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out


# --- Tratamiento de Outliers ---

def aplicar_clipping(df:pd.DataFrame, columna:str, limite:float)->pd.DataFrame:
    df_clipped = df.copy()
    if limite > 0:
        # Límite positivo: actúa como un techo
        df_clipped[columna] = df_clipped[columna].clip(upper=limite)
    else:
        # Límite negativo/cero: actúa como un piso
        df_clipped[columna] = df_clipped[columna].clip(lower=limite)
    
    return df_clipped


def gestiona_outliers(col,clas = 'check'):
     """
     Función para detectar y gestionar outliers en una columna numérica.
     """
    
     #print(col.name)
     # Condición de asimetría y aplicación de criterio 1 según el caso
     if abs(col.skew()) < 1:
        criterio1 = abs((col-col.mean())/col.std())>3
     else:
        criterio1 = abs((col-col.median())/stats.median_abs_deviation(col.dropna()))>6 ## Considerar solo valores válidos!! dropna().
     
     # Calcular primer cuartil     
     q1 = col.quantile(0.25)  
     # Calcular tercer cuartil  
     q3 = col.quantile(0.75)
     # Calculo de IQR
     IQR=q3-q1
     # Calcular criterio 2 (general para cualquier asimetría)
     criterio2 = (col<(q1 - 3*IQR))|(col>(q3 + 3*IQR))
     lower = col[criterio1&criterio2&(col<q1)].count()/col.dropna().count()
     upper = col[criterio1&criterio2&(col>q3)].count()/col.dropna().count()
     # Salida según el tipo deseado
     if clas == 'check':
            return(lower*100,upper*100,(lower+upper)*100)
     elif clas == 'winsor':
            return(soft_winsorize(col,(lower,upper)))
     elif clas == 'miss':
            print('\n MissingAntes: ' + str(col.isna().sum()))
            col.loc[criterio1&criterio2] = np.nan
            print('MissingDespues: ' + str(col.isna().sum()) +'\n')
            return(col)


def winsorize_with_pandas(s, limits):
    """
    Aplica winsorización a una Serie de pandas utilizando los cuantiles como límites.
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))


def soft_winsorize(s, limits):
    """
    Aplica compresión asintótica a los outliers en lugar de un corte duro.
    """
    q_lower = s.quantile(limits[0], interpolation='lower')
    q_upper = s.quantile(1-limits[1], interpolation='higher')
    
    s_mod = s.copy()
    
    # Comprimir cola superior suavemente
    mask_upper = s > q_upper
    s_mod.loc[mask_upper] = q_upper + np.log1p(s.loc[mask_upper] - q_upper)
    
    # Comprimir cola inferior suavemente
    mask_lower = s < q_lower
    s_mod.loc[mask_lower] = q_lower - np.log1p(q_lower - s.loc[mask_lower])
    
    return s_mod

def obtener_cols_flag(df:pd.DataFrame)->list[str]:
    return df.columns[df.columns.str.contains('_IsMissing')].tolist()


# --- Bloque principal ---

# Se replica el flujo del Notebook 02_Transformacion
# Ejecutar desde la raiz: python -m src.transform

def main():
    # --- Preliminares ---

    # Abrir archivo raw_data
    df = pd.read_parquet(raw_data_file)

    # Se asegura el ordenamiento por fecha
    df = df.sort_values(by='Date').reset_index(drop=True)


    # --- Limpieza de datos ---

    # Limpiar cadenas de texto en Sector y Industry
    df_clean = limpiar_industry_y_sector(df)

    # Columnas financieras y volumen expresadas en millones
    df_clean = columnas_en_millones(df_clean)

    # Corrección de anomalías
    df_clean = corregir_anomalias(df_clean)

    print("Limpieza de datos finalizada.")


    # --- Tratamiento Inicial de Missings ---

    # Si se detectan missing  en Industry y Sector:
    #df_clean = recuperar_info(df_clean)
    # Si persistieran casos, imputar manualmente en el modulo clean_transform:
    #df_clean = imputar_info(df_clean)

    # imputar equivalencias financieras
    df_financials_imputed = imputar_equivalencias_financieras(df_clean)

    # Se imputan las columnas financieras, por su media o mediana móvil según sus asimetrías
    df_financials_imputed = imputar_numericas(df_financials_imputed)

    # Se eliminan missings en columnas numéricas relevantes
    cols_no_relevantes = ['GrossProfit', 'FinancingCashFlow', 'InvestingCashFlow']
    df_financials_imputed = quitar_nulos_relevantes(df_financials_imputed, cols_no_relevantes)

    print("Tratamiento Inicial de Missings finalizado.")


    # --- Feature Engineering ---

    # Crear feature YearsSinceAdded
    df_with_features = crear_years_since_added(df_financials_imputed)

    # Calcular métricas financieras y ratios de valuación:
    df_with_features = calcular_metricas(df_with_features)

    # Calcular AverageDailyVolume
    df_with_features = convertir_volumen_a_adv(df_with_features)

    # Aplicar lag de un trimestre a Volume
    columnas_lag1 = ['AverageDailyVolume']
    df_with_features = calcular_lag(df_with_features, columnas_lag1, months=1)

    # Calcular crecimientos
    crecimiento_cols = [
        'TotalRevenue_TTM',
        'EBITDA_TTM',
        'FreeCashFlow_TTM',
        'CapitalExpenditure_TTM',
        'AverageDailyVolume_Lag1'
    ]
    df_with_features = calcular_crecimientos(df_with_features, crecimiento_cols)

    # Antes de calcular las aceleraciones, se imputan crecimientos desconocidos 
    # con tasas de crecimiento neutral (tasa de inflación).
    df_with_features = imputar_crecimientos(df_with_features, crecimiento_cols)

    # Se calculan las aceleraciones
    df_with_features = calcular_aceleraciones(df_with_features, crecimiento_cols)

    # Calcular retornos
    # Se abre el fichero de precios del Índice del Mercado para calcular las covarianzas
    df_index = pd.read_parquet(market_index_file)

    df_with_features = calcular_retornos_y_betas(df_with_features, df_index)  

    print("Feature Engineering finalizado.")


    # --- Tratamiento Final de Missings ---

    # Se aplica la imputación de medias móviles sobre las nuevas variables
    df_imputed = imputar_numericas(df_with_features)

    # Se eliminan missings remanentes en columnas numéricas relevantes
    cols_no_relevantes = []
    df_imputed = quitar_nulos_relevantes(df_imputed, cols_no_relevantes)

    print("Tratamiento de Missings finalizado.")


    # --- Transformaciones Iniciales ---  

    # Transformaciones logarítmicas
    columnas_log = [ 
        'DebtToEquity',
        'CurrentRatio',
        'MarketCap' 
        ]

    df_transformed = transformar_log(
        df_imputed, 
        columnas_log, 
        calculo_1p=True
        )
   
    # Transformaciones Yeo-Johnson
    columnas_yeo = [ 
        'MonthlyReturn',
        'TotalRevenue_TTM_QoQ',
        'FcfToEbitda',
        'TotalRevenue_TTM_YoY',
        'AverageDailyVolume_Lag1_QoQ',
        'AverageDailyVolume_Lag1_YoY',
        'EBITDA_TTM_QoQ',
        'EBITDA_TTM_YoY',
        'OperatingMargins', 
        'ProfitMargins', 
        'ReturnOnAssets', 
        'ReturnOnEquity',
        'EnterpriseValue',
        'Variance'    
        ]

    df_transformed = transformar_yeo_johnson(
        df_transformed, 
        columnas_yeo
        )

    # Agrupar por cuantiles
    cols_a_agrupar = [
        'MonthlyExcessReturn'       
    ]

    # Aplicar la categorización
    for col in cols_a_agrupar:
        df_transformed = categorizar_en_cuantiles(df_transformed, columna=col, num_cuantiles=5)

    print("Transformaciones iniciales finalizadas.")


    # --- Tratamiento Inicial de Outliers ---

    # Recortar valores extremos
    # Columnas a recortar:
    tuplas_clipping = [
        ('EbitdaYield', -2.0),
        ('EbitdaYield', 2.0),
        ('EarningsYield', -2.0),
        ('EarningsYield', 2.0),          
        ('TotalRevenue_TTM_Acceleration', -2.0),
        ('TotalRevenue_TTM_Acceleration', 2.0),
        ('CapExToRevenue', 1.0)
        ] 
    for col, limit in tuplas_clipping:
        df_transformed = aplicar_clipping(df_transformed, columna=col, limite = limit)

    # Definir columnas que saltean la "winsorización"
    cols_fin_clean = obtener_cols_financieras(incluirTTM=True)

    # Columnas flag
    cols_flag = obtener_cols_flag(df_transformed)

    columnas_intactas = cols_fin_clean + cols_flag + [
        'Date', 
        'Ticker',
        'Close',
        'Open',
        'AverageDailyVolume_Lag1'
        ]

    # Separar el dataset
    df_passthrough = df_transformed[columnas_intactas].copy()
    df_transformed_features = df_transformed.drop(columns=columnas_intactas)

    df_cont_transformed = df_transformed_features.select_dtypes(include="number")
    df_winsor = df_cont_transformed.apply(lambda x: gestiona_outliers(x, clas='winsor'))

    print("Gestión inicial de outliers finalizada.")

    # --- Transformaciones Finales ---

    # Logarítmicas
    columnas_log = [ 
        'CapExToRevenue'
        ]

    df_transformed_final = transformar_log(
        df_winsor, 
        columnas_log, 
        calculo_1p=True
        )
    
    # Transformaciones Yeo-Johnson
    columnas_yeo = [ 
        'BookToMarket',
        'EarningsYield'  
        ]

    df_transformed_final = transformar_yeo_johnson(
        df_transformed_final, 
        columnas_yeo
        )
    
    print("Transformaciones finalizadas.")


    # --- Tratamiento Final de Outliers ---

    # Columnas a recortar:
    tuplas_clipping = [
        ('BookToMarket_Yeo', -3.5),
        ('CapExToRevenue_Log1p', 0.45),
        ('FreeCashFlow_TTM_Acceleration', -10.0),
        ('FreeCashFlow_TTM_Acceleration', 10.0),
        ('ShortTermBeta', -2.5),
        ('ShortTermBeta', 5.0),
        ('RevenueYield', -2.5),
        ('RevenueYield', 7.5),
        ('EarningsYield_Yeo', -5.0),
        ('EarningsYield_Yeo', 5.0)
        ] 
    for col, limit in tuplas_clipping:
        df_transformed_final = aplicar_clipping(df_transformed_final, columna=col, limite = limit)

    print("Tratamiento de outliers finalizado.")


    # --- Concatenación final y almacenamiento ---

    df_non_numeric = df_transformed_features.select_dtypes(exclude='number')
    # Se unen variables contínuas transformadas y variables no numéricas
    df_combined = pd.concat([df_non_numeric, df_transformed_final], axis=1)

    # Unir con las columnas que fueron salteadas
    df_final = pd.concat([df_passthrough, df_combined], axis=1)

    # Guardar datos extraidos en fichero clean_data
    # Asegurar que la estructura de directorios exista
    clean_data_file.parent.mkdir(parents=True, exist_ok=True)

    # Guardar el dataframe
    df_final.to_parquet(clean_data_file)
    print(f"Fichero 'clean_data.parquet' guardado en la carpeta de datos.")
    print("Dimensión de datos finales:", df_final.shape)

if __name__ == "__main__":
    main()