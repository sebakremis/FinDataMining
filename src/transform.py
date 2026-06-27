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
from src.clean_transform import corregir_anomalias
from src.config import cols_balance, cols_cashflow, cols_resultados, data_folder

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


def columnas_en_millones(df:pd.DataFrame)->pd.DataFrame:
    cols = obtener_cols_financieras(incluirTTM=False)
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
    return df.isna().mean().sort_values(ascending=False)


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
    df_imputado['DepreciationAndAmortization'] = df_imputado['DepreciationAndAmortization'].fillna(0)

    # Se imputa 'EBITDA' mediante la ecuación: EBITDA = Operating Income + D&A
    cond_falta_ebitda = df_imputado['EBITDA'].isna() & df_imputado['OperatingIncome'].notna()
    df_imputado.loc[cond_falta_ebitda, 'EBITDA'] = (
        df_imputado.loc[cond_falta_ebitda, 'OperatingIncome'] + 
        df_imputado.loc[cond_falta_ebitda, 'DepreciationAndAmortization']
    )

    # =========================================================================
    # 2. IMPUTACIONES DE FLUJO DE CAJA (CASH FLOW)
    # =========================================================================
    
    # Imputar FreeCashFlow (Fórmula: FreeCashFlow = OperatingCashFlow - Capital Expenditure)
    # Se usa .abs() en CapEx para estandarizar salidas de caja y restarlas correctamente
    if 'FreeCashFlow' in df_imputado.columns and 'OperatingCashFlow' in df_imputado.columns and 'CapitalExpenditure' in df_imputado.columns:
        cond_falta_fcf = df_imputado['FreeCashFlow'].isna() & df_imputado['OperatingCashFlow'].notna() & df_imputado['CapitalExpenditure'].notna()
        df_imputado.loc[cond_falta_fcf, 'FreeCashFlow'] = (
            df_imputado.loc[cond_falta_fcf, 'OperatingCashFlow'] - 
            df_imputado.loc[cond_falta_fcf, 'CapitalExpenditure'].abs()
        )

    # =========================================================================
    # 3. IMPUTACIONES DE BALANCE GENERAL (DEUDA)
    # =========================================================================
    
    columnas_deuda = ['TotalDebt', 'CurrentDebt', 'LongTermDebt']
    if not all(col in df_imputado.columns for col in columnas_deuda):
        print("Advertencia: Faltan columnas de deuda. No se realizó imputación de deuda.")
        return df_imputado

    # CASO A: Rescate de TotalDebt (TotalDebt = CurrentDebt + LongTermDebt)
    # Si falta el Total, pero tenemos los dos componentes, se suman
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
    # C1) Falta Corto Plazo
    cond_falta_current = df_imputado['CurrentDebt'].isna() & df_imputado['TotalDebt'].notna() & df_imputado['LongTermDebt'].notna()
    df_imputado.loc[cond_falta_current, 'CurrentDebt'] = (
        df_imputado.loc[cond_falta_current, 'TotalDebt'] - 
        df_imputado.loc[cond_falta_current, 'LongTermDebt']
    )

    # C2) Falta Largo Plazo (LongTermDebt = TotalDebt - CurrentDebt)
    cond_falta_long = df_imputado['LongTermDebt'].isna() & df_imputado['TotalDebt'].notna() & df_imputado['CurrentDebt'].notna()
    df_imputado.loc[cond_falta_long, 'LongTermDebt'] = (
        df_imputado.loc[cond_falta_long, 'TotalDebt'] - 
        df_imputado.loc[cond_falta_long, 'CurrentDebt']
    )

    # SEGURIDAD: Evitar deuda negativa por discrepancias en reportes financieros
    df_imputado['CurrentDebt'] = df_imputado['CurrentDebt'].clip(lower=0)
    df_imputado['LongTermDebt'] = df_imputado['LongTermDebt'].clip(lower=0)

    return df_imputado


def imputar_numericas(df:pd.DataFrame)->pd.DataFrame:
    """
    Aplica una media móvil o mediana móvil a las columnas numéricas de un DataFrame
    dependiendo de su asimetría (skewness).
    """
    # Creamos una copia
    df_resultado = df.copy()

    umbral = 3 # Valor usado para separar las simétricas de las no simétricas
    
    # Se seleccionan las columnas numéricas
    cols_numericas = df_resultado.select_dtypes(include=[np.number]).columns
    
    # Iteramos solo sobre las numéricas
    for col in cols_numericas:
        # Calcular la asimetría
        
        sesgo = df_resultado[col].skew()
        
        # Manejo de casos límite: si hay muy pocos datos, skew() devuelve NaN
        if pd.isna(sesgo):
            continue 
            
        # Se separa según el valor absoluto del umbral, porque la asimetría puede ser negativa
        if abs(sesgo) < umbral:
            # Simétricas: Media móvil (mean)
            df_resultado[col] = df_resultado[col].rolling(window=3, min_periods=1).mean()
        else:
            # Asimétricas: Mediana móvil (median)
            df_resultado[col] = df_resultado[col].rolling(window=3, min_periods=1).median()
            
    # Las columnas no numéricas no se tocan y se devuelven tal cual en el df_resultado
    return df_resultado


def aplicar_interpolacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina TODOS los missings en las columnas numéricas por cada Ticker.
    Conecta puntos intermedios linealmente y proyecta de forma plana (ffill/bfill) 
    en los extremos, sin límite de registros.
    """
    cols = df.select_dtypes(include=np.number).columns
    
    # .transform() asegura que los índices se mantengan alineados con el df original
    df[cols] = df.groupby('Ticker')[cols].transform(
        lambda x: x.interpolate(
            method='linear', 
            limit_direction='both'  # 'both' asegura que rellene hacia adelante y hacia atrás en los extremos
        )
    )
    
    return df


def imputar_transversal(df: pd.DataFrame, cols: list, metric: str = 'median') -> pd.DataFrame:
    """
    Imputa valores nulos en variables usando la métrica del sector en la misma fecha exacta. 
    Si el sector entero no tiene datos, usa la métrica de todo el mercado.
    
    Args:
        df: DataFrame original con las columnas 'Date' y 'Sector'.
        cols: Lista de strings con los nombres de las columnas a imputar.
        metric: 'mean' (promedio) o 'median' (mediana).
    """
    # Trabajar sobre una copia
    df_imputado = df.copy()
    
    # Verificar que las columnas necesarias para agrupar existan
    if 'Sector' not in df_imputado.columns or 'Date' not in df_imputado.columns or 'Ticker' not in df_imputado.columns:
        raise KeyError("El DataFrame debe contener las columnas 'Date', 'Sector' y 'Ticker'.")

    for col in cols:
        if col not in df_imputado.columns:
            continue
            
        # Imputación Primaria: Agrupar por Fecha y Sector
        metrica_sectorial = df_imputado.groupby(['Date', 'Sector'], observed=True)[col].transform(metric)
        df_imputado[col] = df_imputado[col].fillna(metrica_sectorial)
        
        # Imputación Secundaria (Fallback): Por si un sector entero es NaN ese mes
        if df_imputado[col].isnull().any():
            metrica_mercado = df_imputado.groupby('Date')[col].transform(metric)
            df_imputado[col] = df_imputado[col].fillna(metrica_mercado)
            
        # Imputación Terciaria: Si todo el mercado es NaN, llenar con bfill
        if df_imputado[col].isnull().any():
            df_imputado[col] = df_imputado.groupby('Ticker')[col].bfill()

    return df_imputado


# --- Funciones de Feature Engineering ---

def obtener_cols_financieras(incluirTTM:bool=True)->list:
    if incluirTTM:
        cadena = '_TTM'
    else:
        cadena = ''
    
    cols_cashflow_ttm = [col + cadena for col in cols_cashflow]
    cols_resultados_ttm = [col + cadena for col in cols_resultados]
    cols_financieras_raw = cols_balance + cols_cashflow_ttm + cols_resultados_ttm
    cols_financieras = [col.replace(' ', '') for col in cols_financieras_raw]

    return cols_financieras


def transformar_flujos_a_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma variables financieras de flujo a TTM (Trailing Twelve Months) 
    sumando los últimos 4 trimestres. Añade el sufijo '_TTM' y elimina las originales.
    
    Args:
        df (pd.DataFrame): DataFrame original con datos trimestrales.
        cols_income (list): Lista de columnas del Income Statement (ej. TotalRevenue, NetIncome).
        cols_cashflow (list): Lista de columnas del Cash Flow (ej. OperatingCashFlow).
        
    Returns:
        pd.DataFrame: DataFrame con las columnas TTM calculadas y las originales eliminadas.
    """
    # Crear una copia para evitar modificar el DataFrame original por referencia
    df_ttm = df.copy()
    
    # Unir las listas de variables de flujo
    cols_flujo_raw = cols_resultados + cols_cashflow
    cols_flujo = [col.replace(' ', '') for col in cols_flujo_raw]
    
    # Filtrar para operar sólo sobre las columnas que realmente existen en el DataFrame
    cols_presentes = [col for col in cols_flujo if col in df_ttm.columns]
    
    # Ordenar por Ticker y Date para que el rolling sea cronológico
    df_ttm = df_ttm.sort_values(by=['Ticker', 'Date'])
    
    # CALCULAR TTM: Agrupar por Ticker y aplicar suma móvil de 4 trimestres
    for col in cols_presentes:
        nuevo_nombre = f"{col}_TTM"
        if col == 'BasicAverageShares': 
            # se calcula el promedio en lugar de la suma
            df_ttm[nuevo_nombre] = df_ttm.groupby('Ticker')[col].transform(
                lambda x: x.rolling(window=4, min_periods=2).mean() # Al ser un promedio y poco volátil, se puede reducir la ventana
            )
        else:
            # el resto se calcula la suma
            df_ttm[nuevo_nombre] = df_ttm.groupby('Ticker')[col].transform(
                lambda x: x.rolling(window=4, min_periods=4).sum() # Aqui debe ser el min_periods=4 para que sea TTM
            )
        
    # 3. LIMPIAR: Descartar las columnas originales de flujo
    df_ttm = df_ttm.drop(columns=cols_presentes)
    
    # Restaurar el orden original del índice por consistencia
    df_ttm = df_ttm.sort_index()
    
    return df_ttm


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
    Recibe el DataFrame limpio de precios trimestrales y datos fundamentales alineados, 
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

    # Calcular Capitalización Bursátil expresada en millones (se convirtió previamente BasicAverageShare a millones)
    df_metrics['MarketCap'] = (df_metrics['Open'] * df_metrics['BasicAverageShares_TTM'])

    # Preparar deuda y efectivo para el EnterpriseValue = MarketCap + Deuda Total - Efectivo
    deuda_total = df_metrics['TotalDebt'].fillna(
        df_metrics['CurrentDebt'].fillna(0) + df_metrics['LongTermDebt'].fillna(0)
    )
    efectivo = df_metrics['CashAndCashEquivalents'].fillna(0)
    df_metrics['EnterpriseValue'] = df_metrics['MarketCap'] + deuda_total - efectivo

    # Ratios de valuación
    df_metrics['TrailingPE'] = df_metrics['MarketCap'] / df_metrics['NetIncome_TTM']
    df_metrics['EnterpriseToEbitda'] = df_metrics['EnterpriseValue'] / df_metrics['EBITDA_TTM']
    df_metrics['PriceToBook'] = df_metrics['MarketCap'] / df_metrics['StockholdersEquity']

    # Ratios de rentabilidad y márgenes
    df_metrics['OperatingMargins'] = df_metrics['OperatingIncome_TTM'] / df_metrics['TotalRevenue_TTM']
    df_metrics['ProfitMargins'] = df_metrics['NetIncome_TTM'] / df_metrics['TotalRevenue_TTM']
    df_metrics['ReturnOnEquity'] = df_metrics['NetIncome_TTM'] / df_metrics['StockholdersEquity']
    df_metrics['ReturnOnAssets'] = df_metrics['NetIncome_TTM'] / df_metrics['TotalAssets']

    # Ratios de liquidez y solvencia
    df_metrics['DebtToEquity'] = deuda_total / df_metrics['StockholdersEquity']
    df_metrics['CurrentRatio'] = df_metrics['CurrentAssets'] / df_metrics['CurrentLiabilities']
       
    # Otras métricas
    # Apalancamiento 
    df_metrics['NetDebtToEbitda'] = (deuda_total - df_metrics['CashAndCashEquivalents']) / df_metrics['EBITDA_TTM']

    # Free Cash Flow Conversion
    df_metrics['FcfToEbitda'] = df_metrics['FreeCashFlow_TTM'] / df_metrics['EBITDA_TTM']

    # Capital Intensity
    # Se usa np.abs porque el Capex suele reportarse en negativo en los estados de flujo de caja
    df_metrics['CapExToRevenue'] = np.abs(df_metrics['CapitalExpenditure_TTM']) / df_metrics['TotalRevenue_TTM']
    
    # Reemplazar por NaN todos los infinitos generados por divisiones por cero
    df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forzar NaN en métricas dependientes del Patrimonio si este es negativo
    mask_patrimonio_invalido = df_metrics['StockholdersEquity'] <= 0
    cols_patrimonio = ['PriceToBook', 'ReturnOnEquity', 'DebtToEquity']
    
    # Aplicamos el filtro usando .loc para modificar las columnas específicas
    df_metrics.loc[mask_patrimonio_invalido, cols_patrimonio] = np.nan

    # Forzar NaN en métricas dependientes de Income si son cero o negativos
    mask_netincome_invalido = df_metrics['NetIncome_TTM'] <= 0
    df_metrics.loc[mask_netincome_invalido,'TrailingPE'] = np.nan

    mask_ebitda_invalido = df_metrics['EBITDA_TTM'] <= 0
    df_metrics.loc[mask_ebitda_invalido,'EnterpriseToEbitda'] = np.nan

    # Limpieza final: Redondear columnas
    cols_a_redondear = [
        'TrailingPE', 'PriceToBook', 'EnterpriseToEbitda', 'OperatingMargins', 
        'ProfitMargins', 'ReturnOnEquity', 'ReturnOnAssets', 'DebtToEquity', 'CurrentRatio'
    ]
    
    df_metrics[cols_a_redondear] = df_metrics[cols_a_redondear].round(6) # 6 decimales

    return df_metrics


def convertir_volumen_a_adv(df:pd.DataFrame)->pd.DataFrame:
    df['AverageDailyVolume'] = df['Volume'] / 63 # estimado de 63 días hábiles por trimestre
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
        prev_rev_4 = df_out.groupby('Ticker')[col].shift(4)
        df_out[f'{col}_YoY'] = (df_out[col] - prev_rev_4) / prev_rev_4.abs()
        prev_rev_1 = df_out.groupby('Ticker')[col].shift(1)
        df_out[f'{col}_QoQ'] = (df_out[col] - prev_rev_1) / prev_rev_1.abs()

        # Acotar si quedan resultados absurdos que puedan quedar de dividir por números pequeños
        df_out[f'{col}_YoY'] = df_out[f'{col}_YoY'].clip(lower=-10.0, upper=10.0) # Acota entre -1000% y +1000%
        df_out[f'{col}_QoQ'] = df_out[f'{col}_QoQ'].clip(lower=-10.0, upper=10.0)

    # Reemplazar por NaN todos los infinitos generados por divisiones por cero
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

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


def calcular_retornos(df: pd.DataFrame, df_index: pd.DataFrame, ventana: int = 4, min_periodos: int = 2) -> pd.DataFrame:
    """
    Calcula retornos, varianza del activo y covarianza con el mercado.
    Aplica un rezago (lag) de 1 periodo para evitar Data Leakage.
    """       
    ticker_mercado = df_index['Ticker'].iloc[0]
    
    # Preparar datos y calcular retornos
    df_unido = pd.concat([df, df_index], ignore_index=True)
    df_pivot = df_unido.pivot(index='Date', columns='Ticker', values='Open').sort_index()
    
    # pct_change usa (t / t-1) - 1
    df_retornos = df_pivot.pct_change(fill_method=None)
    
    retornos_mercado = df_retornos[ticker_mercado]
    df_activos = df_retornos.drop(columns=[ticker_mercado])
    
    # Calcular estadísticas móviles
    varianzas_activos = df_activos.rolling(window=ventana, min_periods=min_periodos).var()
    covarianzas = df_activos.rolling(window=ventana, min_periods=min_periodos).cov(retornos_mercado)
    
    # Aplicar lag de 1 periodo (Lo que pasó en t, se mueve a t+1 para ser usado como Feature)
    df_activos_lag = df_activos.shift(1)
    varianzas_activos_lag = varianzas_activos.shift(1)
    covarianzas_lag = covarianzas.shift(1)
    
    # Transformar cada matriz a formato largo (melt) usando los DataFrames rezagados
    df_ret_long = df_activos_lag.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='QuarterlyReturn_Lag1'
    )
    df_var_long = varianzas_activos_lag.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='QuarterlyVariance_Lag1'
    )
    df_cov_long = covarianzas_lag.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='MarketCovariance_Lag1'
    )
    
    # Consolidar todas las features en un solo DataFrame
    df_features = df_ret_long.merge(df_var_long, on=['Date', 'Ticker'])
    df_features = df_features.merge(df_cov_long, on=['Date', 'Ticker'])
    
    # Unir las features con el DataFrame original
    df_final = pd.merge(df, df_features, on=['Date', 'Ticker'], how='left')
    
    return df_final


def calcular_lag(df:pd.DataFrame, cols:list,q:int=1)->pd.DataFrame:
    for col in cols:
        df[col+f'_Lag{q}'] = df[col].shift(4)

    df.drop(columns=cols, inplace=True)
    return df


def calcular_relative_size(df: pd.DataFrame) -> pd.DataFrame:
    # Limpiar anomalías: forzamos que el piso de ingresos y activos sea 0
    revenue_clean = df['TotalRevenue_TTM'].clip(lower=0)
    assets_clean = df['TotalAssets'].clip(lower=0)

    # Agrupar y calcular la suma total del mercado por fecha usando los datos limpios
    df['TotalMarketAssets'] = assets_clean.groupby(df['Date']).transform('sum')
    df['TotalMarketRevenue'] = revenue_clean.groupby(df['Date']).transform('sum')
    
    # Dividir los valores individuales por el total del mercado
    df['RelativeAssets'] = assets_clean / df['TotalMarketAssets']
    df['RelativeRevenue'] = revenue_clean / df['TotalMarketRevenue']

    # Se eliminan las columnas de totales
    df.drop(columns=['TotalMarketAssets', 'TotalMarketRevenue'], inplace=True)
   
    return df


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


def cat_plot(col):
     """
     Gráfico de barras para variables categóricas.
     """
     if col.dtypes == 'category':
        fig = px.bar(col.value_counts())
        #fig = sns.countplot(x=col)
        return(fig)


def plot(col):
     """
     Función general para aplicar al archivo por columnas, detectando el tipo de variable y aplicando el gráfico adecuado.
     """
     if col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_plot(col).show()


# --- Funciones de Transformación

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
            return(winsorize_with_pandas(col,(lower,upper)))
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


# --- Bloque principal ---

# Se replica el flujo del Notebook 02_Transformacion
# Ejecutar desde la raiz: python -m src.transform

def main():
    # --- Preliminares ---

    # Abrir archivo raw_data
    df = pd.read_parquet(f"{data_folder}/raw_data.parquet")

    # Se asegura el ordenamiento por fecha
    df = df.sort_values(by='Date').reset_index(drop=True)

    # --- Limpieza de datos ---

    # Limpiar cadenas de texto en Sector y Industry
    df['Industry'] = df['Industry'].apply(limpiar_cadenas)
    df['Sector'] = df['Sector'].apply(limpiar_cadenas)

    # Convertir Sector y Industry a tipo category
    df['Sector'] = df['Sector'].astype('category')
    df['Industry'] = df['Industry'].astype('category')

    # Columnas financieras y volumen expresadas en millones
    df = columnas_en_millones(df)

    # Corrección de anomalías
    df_clean = corregir_anomalias(df)

    print("Limpieza de datos finalizada.")

    # --- Tratamiento Inicial de Missings ---

    # Se imputa el missing en Industry y Sector
    condicion = df_clean['Ticker'] == 'MKSI'
    df_clean.loc[condicion, 'Sector'] = 'Technology'
    df_clean.loc[condicion, 'Industry'] = 'Scientific And Technical Instruments'

    # imputar equivalencias financieras
    df_fin_imputed = imputar_equivalencias_financieras(df_clean)

    # Se imputan las columnas financieras, por su media o mediana móvil según sus asimetrías
    df_fin_imputed = imputar_numericas(df_fin_imputed)


    # Forward fill y Back fill
    df_fin_imputed = aplicar_interpolacion(df_fin_imputed)

    print("Tratamiento Inicial de Missings finalizado.")

    # --- Feature Engineering ---

    # Variables TTM
    df_with_features = transformar_flujos_a_ttm(df_fin_imputed)

    # Crear feature YearsSinceAdded
    df_with_features = crear_years_since_added(df_with_features)

    # Calcular métricas financieras y ratios de valuación:
    df_with_features = calcular_metricas(df_with_features)

    # Calcular AverageDailyVolume
    df_with_features = convertir_volumen_a_adv(df_with_features)

    # Aplicar lag de un trimestre a Volume
    columnas_lag1 = ['AverageDailyVolume']
    df_with_features = calcular_lag(df_with_features, columnas_lag1, q=1)

    # Calcular crecimientos y aceleraciones
    crecimiento_cols = [
        'TotalRevenue_TTM',
        'EBITDA_TTM',
        'FreeCashFlow_TTM',
        'CapitalExpenditure_TTM',
        'AverageDailyVolume_Lag1'
    ]
    df_with_features = calcular_crecimientos(df_with_features, crecimiento_cols)
    df_with_features = calcular_aceleraciones(df_with_features, crecimiento_cols)

    # Calcular retornos
    # Se abre el fichero de precios del Índice del Mercado para calcular las covarianzas
    df_index = pd.read_parquet(f"{data_folder}/market_index.parquet")

    df_with_features = calcular_retornos(df_with_features, df_index)  

    # Calcular tamaños relativos: RelativeAssets y RelativeRevenue
    df_with_features = calcular_relative_size(df_with_features)

    print("Feature Engineering finalizado.")

    # --- Tratamiento Final de Missings ---

    #Se aplica imputación transversal para las columnas de crecimiento:
    df_imputed = imputar_transversal(df_with_features, crecimiento_cols)

    # Se aplica la imputación de medias móviles sobre las nuevas variables
    df_imputed = imputar_numericas(df_imputed)

    # Se aplican los fills sobre los missings que queden
    df_imputed = aplicar_interpolacion(df_imputed)

    print("Tratamiento de Missings finalizado.")

    # --- Transformaciones ---
   
    # Transformaciones logarítmicas
    columnas_a_transformar = [ 
        'CapExToRevenue',
        'DebtToEquity',
        'QuarterlyVariance_Lag1',
        'MarketCap',
        'EnterpriseValue'
        ]

    df_transformed = transformar_log(
        df_imputed, 
        columnas_a_transformar, 
        calculo_1p=True
        )
    
    print("Transformaciones finalizadas.")

    # --- Tratamiento de Outliers ---

    # Definir columnas que saltean la "winsorización"
    cols_fin_clean = obtener_cols_financieras(incluirTTM=True)

    columnas_intactas = cols_fin_clean + [
        # Variables de precio y ratios
        'Close',
        'Open',    
        'TrailingPE',
        'EnterpriseToEbitda',
        'PriceToBook',
        # Otras
        'Date', 
        'Ticker'     
        ]

    # Separar el dataset
    df_passthrough = df_transformed[columnas_intactas].copy()
    df_transformed_features = df_transformed.drop(columns=columnas_intactas)

    df_cont_transformed = df_transformed_features.select_dtypes(include="number")
    df_winsor = df_cont_transformed.apply(lambda x: gestiona_outliers(x, clas='winsor'))

    print("Gestión de outliers finalizada.")

    # --- Concatenación final y almacenamiento ---

    df_non_numeric_transformed = df_transformed_features.select_dtypes(exclude='number')
    # Se unen variables contínuas transformadas y variables no numéricas
    df_combined = pd.concat([df_non_numeric_transformed, df_winsor], axis=1)
   
    # Unir con las columnas que fueron salteadas
    df_final = pd.concat([df_passthrough, df_combined], axis=1)

    # Guardar datos extraidos en fichero clean_data
    df_final.to_parquet(f"{data_folder}/clean_data.parquet")
    print(f"Transformación finalizada.\nFichero 'clean_data.parquet' guardado en la carpeta {data_folder}.")
    print("Dimensión de datos finales:", df_final.shape)

if __name__ == "__main__":
    main()