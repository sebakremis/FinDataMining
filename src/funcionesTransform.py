"""
src/funcionesTransform.py
Módulo de funciones para la fase de Transformación de Datos (Transform).
"""

import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from src.config import cols_balance, cols_cashflow, cols_resultados

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
        # Usamos transform() para calcular y asignar manteniendo la estructura del índice
        df_ttm[nuevo_nombre] = df_ttm.groupby('Ticker')[col].transform(
            lambda x: x.rolling(window=4, min_periods=4).sum()
        )
        
    # 3. LIMPIAR: Descartar las columnas originales de flujo
    df_ttm = df_ttm.drop(columns=cols_presentes)
    
    # Restaurar el orden original del índice por consistencia
    df_ttm = df_ttm.sort_index()
    
    return df_ttm

def obtener_cols_financieras()->list:
    cols_cashflow_ttm = [col+'_TTM' for col in cols_cashflow]
    cols_resultados_ttm = [col+'_TTM' for col in cols_resultados]
    cols_financieras_raw = cols_balance + cols_cashflow_ttm + cols_resultados_ttm
    cols_financieras = [col.replace(' ', '') for col in cols_financieras_raw]

    return cols_financieras


def imputar_deuda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores nulos en CurrentDebt y LongTermDebt basándose en la 
    relación contable con TotalDebt.
    """
    # Trabajar sobre una copia
    df_imputado = df.copy()
    
    # Verificar que las columnas existan antes de operar
    columnas_deuda = ['TotalDebt', 'CurrentDebt', 'LongTermDebt']
    if not all(col in df_imputado.columns for col in columnas_deuda):
        print("Advertencia: Faltan columnas de deuda. No se realizó imputación.")
        return df_imputado

    # CASO 1: Si TotalDebt es 0, las componentes son 0
    # Usamos fillna(False) por si TotalDebt también es NaN en algunas filas
    condicion_cero = (df_imputado['TotalDebt'] == 0).fillna(False)
    
    df_imputado.loc[condicion_cero & df_imputado['CurrentDebt'].isnull(), 'CurrentDebt'] = 0
    df_imputado.loc[condicion_cero & df_imputado['LongTermDebt'].isnull(), 'LongTermDebt'] = 0

    # CASO 2: Deducción por resta (Total = Corto + Largo)
    # A) Falta Corto Plazo
    cond_falta_current = df_imputado['CurrentDebt'].isnull() & \
                         df_imputado['TotalDebt'].notnull() & \
                         df_imputado['LongTermDebt'].notnull()
    
    df_imputado.loc[cond_falta_current, 'CurrentDebt'] = \
        df_imputado.loc[cond_falta_current, 'TotalDebt'] - df_imputado.loc[cond_falta_current, 'LongTermDebt']

    # B) Falta Largo Plazo
    cond_falta_long = df_imputado['LongTermDebt'].isnull() & \
                      df_imputado['TotalDebt'].notnull() & \
                      df_imputado['CurrentDebt'].notnull()
    
    df_imputado.loc[cond_falta_long, 'LongTermDebt'] = \
        df_imputado.loc[cond_falta_long, 'TotalDebt'] - df_imputado.loc[cond_falta_long, 'CurrentDebt']

    # SEGURIDAD: Evitar deuda negativa por discrepancias en reportes financieros
    # (A veces las APIs tienen desajustes de centavos o datos corruptos)
    df_imputado['CurrentDebt'] = df_imputado['CurrentDebt'].clip(lower=0)
    df_imputado['LongTermDebt'] = df_imputado['LongTermDebt'].clip(lower=0)

    return df_imputado

# Cálculos de métricas financieras

def calcular_metricas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Recibe el DataFrame limpio de precios MENSUALES y datos fundamentales alineados, 
    y calcula métricas financieras históricas.
    """
    # Definir columnas necesarias para calcular
    cols_necesarias = [
        'Ticker', 
        'Date', 
        'Close', 
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

    # Calcular Capitalización Bursátil
    df_metrics['MarketCap'] = df_metrics['Close'] * df_metrics['BasicAverageShares_TTM']

    # Preparar deuda y efectivo para el EnterpriseValue = MarketCap + Deuda Total - Efectivo
    deuda_total = df_metrics['TotalDebt'].fillna(
        df_metrics['CurrentDebt'].fillna(0) + df_metrics['LongTermDebt'].fillna(0)
    )
    efectivo = df_metrics['CashAndCashEquivalents'].fillna(0)
    df_metrics['EnterpriseValue'] = df_metrics['MarketCap'] + deuda_total - efectivo

    # Se define un valor pequeño "epsilon" para evitar divisiones por cero
    epsilon = 1e-6

    # Ratios de valuación
    df_metrics['TrailingPE'] = df_metrics['MarketCap'] / (df_metrics['NetIncome_TTM'] + epsilon)
    df_metrics['EnterpriseToEbitda'] = df_metrics['EnterpriseValue'] / (df_metrics['EBITDA_TTM'] + epsilon)
    df_metrics['PriceToBook'] = df_metrics['MarketCap'] / (df_metrics['StockholdersEquity'] + epsilon)

    # Ratios de rentabilidad y márgenes
    df_metrics['OperatingMargins'] = df_metrics['OperatingIncome_TTM'] / (df_metrics['TotalRevenue_TTM'] + epsilon)
    df_metrics['ProfitMargins'] = df_metrics['NetIncome_TTM'] / (df_metrics['TotalRevenue_TTM'] + epsilon)
    df_metrics['ReturnOnEquity'] = df_metrics['NetIncome_TTM'] / (df_metrics['StockholdersEquity'] + epsilon)
    df_metrics['ReturnOnAssets'] = df_metrics['NetIncome_TTM'] / (df_metrics['TotalAssets'] + epsilon)

    # Ratios de liquidez y solvencia
    df_metrics['DebtToEquity'] = deuda_total / (df_metrics['StockholdersEquity'] + epsilon)
    df_metrics['CurrentRatio'] = df_metrics['CurrentAssets'] / (df_metrics['CurrentLiabilities'] + epsilon)
       
    # Ratios de crecimiento
    '''
    - Crecimiento interanual (Year-over-Year, YoY) - Ventana de 12 meses y Trimestral (QoQ)
    - Se aplica .abs() en el denominador para corregir matemáticamente la dirección del 
    crecimiento si el período anterior era negativo.
    - No me decido si utilizar una función wrapper, me parece más simple dejarlo así.
    '''
    
    crecimiento_cols = []

    # Revenue Growth
    prev_rev_12 = df_metrics.groupby('Ticker')['TotalRevenue_TTM'].shift(12)
    df_metrics['Revenue_YoY'] = (df_metrics['TotalRevenue_TTM'] - prev_rev_12) / (prev_rev_12.abs() + epsilon)
    prev_rev_3 = df_metrics.groupby('Ticker')['TotalRevenue_TTM'].shift(3)
    df_metrics['Revenue_QoQ'] = (df_metrics['TotalRevenue_TTM'] - prev_rev_3) / (prev_rev_3.abs() + epsilon)
    crecimiento_cols.extend(['Revenue_YoY', 'Revenue_QoQ'])
    
    # EBITDA Growth
    prev_ebitda_12 = df_metrics.groupby('Ticker')['EBITDA_TTM'].shift(12)
    df_metrics['Ebitda_YoY'] = (df_metrics['EBITDA_TTM'] - prev_ebitda_12) / (prev_ebitda_12.abs() + epsilon)
    prev_ebitda_3 = df_metrics.groupby('Ticker')['EBITDA_TTM'].shift(3)
    df_metrics['Ebitda_QoQ'] = (df_metrics['EBITDA_TTM'] - prev_ebitda_3) / (prev_ebitda_3.abs() + epsilon)
    crecimiento_cols.extend(['Ebitda_YoY', 'Ebitda_QoQ'])
    
    # Free Cash Flow Growth
    prev_FCF_12 = df_metrics.groupby('Ticker')['FreeCashFlow_TTM'].shift(12)
    df_metrics['Fcf_YoY'] = (df_metrics['FreeCashFlow_TTM'] - prev_FCF_12) / (prev_FCF_12.abs() + epsilon)
    prev_FCF_3 = df_metrics.groupby('Ticker')['FreeCashFlow_TTM'].shift(3)
    df_metrics['Fcf_QoQ'] = (df_metrics['FreeCashFlow_TTM'] - prev_FCF_3) / (prev_FCF_3.abs() + epsilon)
    crecimiento_cols.extend(['Fcf_YoY', 'Fcf_QoQ'])

    # CapEx Growth
    prev_Capex_12 = df_metrics.groupby('Ticker')['CapitalExpenditure_TTM'].shift(12)
    df_metrics['CapEx_YoY'] = (df_metrics['CapitalExpenditure_TTM'] - prev_Capex_12) / (prev_Capex_12.abs() + epsilon)
    prev_Capex_3 = df_metrics.groupby('Ticker')['CapitalExpenditure_TTM'].shift(3)
    df_metrics['CapEx_QoQ'] = (df_metrics['CapitalExpenditure_TTM'] - prev_Capex_3) / (prev_Capex_3.abs() + epsilon)
    crecimiento_cols.extend(['CapEx_YoY', 'CapEx_QoQ'])

    # Otras métricas
    # Apalancamiento 
    df_metrics['NetDebtToEbitda'] = (deuda_total - df_metrics['CashAndCashEquivalents']) / (df_metrics['EBITDA_TTM'] + epsilon)

    # Free Cash Flow Conversion
    df_metrics['FcfToEbitda'] = df_metrics['FreeCashFlow_TTM'] / (df_metrics['EBITDA_TTM'] + epsilon)

    # Capital Intensity
    # Se usa np.abs porque el Capex suele reportarse en negativo en los estados de flujo de caja
    df_metrics['CapExToRevenue'] = np.abs(df_metrics['CapitalExpenditure_TTM']) / (df_metrics['TotalRevenue_TTM'] + epsilon)
    
    # Limpieza de posibles infinitos creados por EBITDAs muy cercanos a cero
    otras_cols = ['NetDebtToEbitda', 'FcfToEbitda', 'CapExToRevenue', 'TrailingPE']
    cols_a_limpiar = crecimiento_cols + otras_cols
    df_metrics[cols_a_limpiar] = df_metrics[cols_a_limpiar].replace([np.inf, -np.inf], np.nan)

    # Acotar si quedan resultados absurdos que puedan quedar de dividir por números pequeños
    for col in crecimiento_cols:
            df_metrics[col] = df_metrics[col].clip(lower=-10.0, upper=10.0) # Acota entre -1000% y +1000%

    # Limpieza final: Redondear columnas
    cols_a_redondear = [
        'TrailingPE', 'PriceToBook', 'EnterpriseToEbitda', 'OperatingMargins', 
        'ProfitMargins', 'ReturnOnEquity', 'ReturnOnAssets', 'DebtToEquity', 'CurrentRatio'
    ]
    
    df_metrics[cols_a_redondear] = df_metrics[cols_a_redondear].round(6) # 6 decimales

    return df_metrics, crecimiento_cols


def calcular_retornos(df: pd.DataFrame, df_index: pd.DataFrame, ventana: int = 4, min_periodos: int = 2) -> pd.DataFrame:
    """
    Calcula retornos mensuales, varianza del activo y covarianza con el mercado.
    """       
    ticker_mercado = df_index['Ticker'].iloc[0]
    
    # Preparar datos y calcular retornos
    df_unido = pd.concat([df, df_index], ignore_index=True)
    df_pivot = df_unido.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    df_retornos = df_pivot.pct_change(fill_method=None).dropna(how='all')
    
    retornos_mercado = df_retornos[ticker_mercado]
    df_activos = df_retornos.drop(columns=[ticker_mercado])
    
    # Calcular estadísticas móviles (Features)
    varianzas_activos = df_activos.rolling(window=ventana, min_periods=min_periodos).var()
    covarianzas = df_activos.rolling(window=ventana, min_periods=min_periodos).cov(retornos_mercado)
    
    # Transformar cada matriz a formato largo (melt)
    df_ret_long = df_activos.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='QuarterlyReturn'
    )
    df_var_long = varianzas_activos.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='QuarterlyVariance'
    )
    df_cov_long = covarianzas.reset_index().melt(
        id_vars='Date', var_name='Ticker', value_name='MarketCovariance'
    )
    
    # Consolidar todas las features en un solo DataFrame
    df_features = df_ret_long.merge(df_var_long, on=['Date', 'Ticker'])
    df_features = df_features.merge(df_cov_long, on=['Date', 'Ticker'])
    
    # Limpiar registros sin suficientes datos (NaNs de la ventana inicial)
    df_features = df_features.dropna(subset=['QuarterlyVariance', 'MarketCovariance']).reset_index(drop=True)
    
    # Redondear para mayor legibilidad
    df_features = df_features.round(6)

    # Unir las features con el DataFrame original
    df_final = pd.merge(df, df_features, on=['Date', 'Ticker'], how='left')
    
    return df_final


def imputar_transversal(df: pd.DataFrame, cols: list, metric: str = 'median') -> pd.DataFrame:
    """
    Imputa valores nulos en variables usando la métrica del sector en la misma fecha exacta. 
    Si el sector entero no tiene datos, usa la métrica de todo el mercado.
    
    Args:
        df: DataFrame original con las columnas 'Date' y 'Sector'.
        cols: Lista de strings con los nombres de las columnas a imputar.
        metric: 'mean' (promedio) o 'median' (mediana). Se recomienda 'median' en finanzas.
    """
    # Trabajar sobre una copia
    df_imputado = df.copy()
    
    # Verificar que las columnas necesarias para agrupar existan
    if 'Sector' not in df_imputado.columns or 'Date' not in df_imputado.columns:
        raise KeyError("El DataFrame debe contener las columnas 'Date' y 'Sector'.")

    for col in cols:
        if col not in df_imputado.columns:
            continue
            
        # Imputación Primaria: Agrupar por Fecha y Sector
        metrica_sectorial = df_imputado.groupby(['Date', 'Sector'])[col].transform(metric)
        df_imputado[col] = df_imputado[col].fillna(metrica_sectorial)
        
        # Imputación Secundaria (Fallback): Por si un sector entero es NaN ese mes
        if df_imputado[col].isnull().any():
            metrica_mercado = df_imputado.groupby('Date')[col].transform(metric)
            df_imputado[col] = df_imputado[col].fillna(metrica_mercado)
            
        # Imputación Terciaria: Si todo el mercado es NaN, llenar con bfill
        if df_imputado[col].isnull().any():
            df_imputado[col] = df_imputado.groupby('Ticker')[col].bfill()

    return df_imputado


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
   
    return df


def gestiona_outliers(col,clas = 'check'):
     """
     Función para detectar y gestionar outliers en una columna numérica.
     """
    
     print(col.name)
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

# Gráficos

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


# Bloque principal para pruebas desde el terminal

def main():
    pass

if __name__ == "__main__":
    main()