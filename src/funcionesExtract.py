"""
src/funcionesExtract.py
Módulo de funciones para la fase de Ingestión de Datos (Extract).
"""
import pandas as pd
import numpy as np
import yfinance as yf
from src.config import periodo, intervalo, cols_resultados, cols_balance, cols_cashflow
from datetime import datetime
import urllib.request
import os

def descargar_constituents(data_folder="data", force_update=False):
    """
    Descarga el listado de componentes del S&P 500 si no existe o si se fuerza la actualización.
    """
    file_path = f"{data_folder}/constituents.csv"
    url_raw = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    
    os.makedirs(data_folder, exist_ok=True)
    
    if not os.path.exists(file_path) or force_update:
        print("Descargando constituents.csv desde GitHub...")
        urllib.request.urlretrieve(url_raw, file_path)
        print("Descarga completada.")
    else:
        print("Usando archivo constituents.csv local.")
        
    return file_path

def extraer_precios(tickers_list: list) -> pd.DataFrame:
    """
    Extrae precios históricos y formatea las fechas para cruzar con datos fundamentales.
    """
    dfs_prices = []
    for ticker in tickers_list:
        df = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
        
        if df.empty:
            print(f"Sin datos de precio para {ticker}")
            continue
            
        df['Ticker'] = ticker
        
        # 1. Convertir el índice 'Date' en una columna
        df = df.reset_index()
        dfs_prices.append(df)  
        
    if not dfs_prices:
        return pd.DataFrame()

    df_prices = pd.concat(dfs_prices, ignore_index=True)
    
    # 2. Eliminar columnas innecesarias (usar errors='ignore' por si no hay splits/dividends)
    df_prices.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')

    # 3. Renombrar 'Date' a 'Fecha' y quitar la zona horaria para emparejar con fundamentales
    df_prices = df_prices.rename(columns={'Date': 'Fecha'})
    df_prices['Fecha'] = pd.to_datetime(df_prices['Fecha']).dt.tz_localize(None)

    return df_prices


def extraer_financials(tickers_list: list) -> pd.DataFrame:
    """
    Extrae datos financieros del Estado de Resultados, Balance General y Cash Flow,
    devuelve un DataFrame unificado para cálculo de ratios históricos.
    """
    dfs_lista = []       

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Extraer reportes anuales
            fin = yf_ticker.financials
            bal = yf_ticker.balance_sheet
            cf = yf_ticker.cashflow

            # Validación del Estado de Resultados
            if fin is None or fin.empty:
                print(f"Sin datos financieros para {ticker}")
                continue

            # Transponer y filtrar Estado de Resultados
            df_fin = fin.T.reindex(columns=cols_resultados)

            # Validación del Balance General (si existe)
            if bal is not None and not bal.empty:
                df_bal = bal.T.reindex(columns=cols_balance)
                df_temp = df_fin.join(df_bal, how='left')
            else:
                # Si no hay balance, nos quedamos con financials y llenamos el resto con nulos
                df_temp = df_fin.copy() # Usamos .copy() para evitar SettingWithCopyWarning
                for col in cols_balance:
                    df_temp[col] = float('nan') # Usar NaN nativo en lugar de pd.NA
                
                # Forzar que estas nuevas columnas nulas sean reconocidas como numéricas (float)
                df_temp[cols_balance] = df_temp[cols_balance].astype(float)  

            # Validación del Cash Flow (si existe)
            if cf is not None and not cf.empty:
                df_cf = cf.T.reindex(columns=cols_cashflow)
                # Unimos a df_temp usando el índice (Fecha)
                df_temp = df_temp.join(df_cf, how='left')
            else:
                # Fallback si no hay datos de Cash Flow
                for col in cols_cashflow:
                    df_temp[col] = float('nan')
                df_temp[cols_cashflow] = df_temp[cols_cashflow].astype(float)        

            # Limpieza del DataFrame temporal
            df_temp = df_temp.reset_index() # Pasamos la fecha del índice a una columna
            df_temp = df_temp.rename(columns={'index': 'Fecha'})
            df_temp['Ticker'] = ticker # Identificador del activo

            # --- TRANSFORMACIÓN DE FECHAS (Regla de 60 días) ---
            # Para evitar "Lookahead Bias", asumimos que la información fue pública 60 días después del cierre.
            # Convertimos a datetime, sumamos los 60 días, y extraemos la fecha pura (.dt.date)
            fechas_datetime = pd.to_datetime(df_temp['Fecha']).dt.tz_localize(None)
            df_temp['Fecha'] = (fechas_datetime + pd.Timedelta(days=60)).dt.date

            # Añadir a la lista solo si no está vacío
            if not df_temp.empty:
                dfs_lista.append(df_temp)

        except Exception as e:
            print(f"Error procesando fundamentales para {ticker}: {e}")
            continue

    # Concatenación final
    if dfs_lista:
        # ignore_index=True es clave aquí para tener un índice numérico limpio (0, 1, 2...)
        df_final = pd.concat(dfs_lista, axis=0, ignore_index=True)
        return df_final
    else:
        return pd.DataFrame()


# Cálculos de métricas financieras y ratios de valuación

def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame limpio de precios y fundamentales alineados, 
    y calcula métricas de valoración históricas.
    """
    # Trabajamos sobre una copia para no alterar el original inadvertidamente
    df_metrics = df.copy()

    # Datos Base
    if 'Basic Average Shares' in df_metrics.columns:
        df_metrics['MarketCap'] = df_metrics['Close'] * df_metrics['Basic Average Shares']
    else:
        print("Advertencia: Falta 'Basic Average Shares'. No se calcularán métricas de mercado.")
        return df_metrics

    # Preparar deuda y efectivo para EV
    deuda_total = df_metrics['Total Debt'].fillna(
        df_metrics['Current Debt'].fillna(0) + df_metrics['Long Term Debt'].fillna(0)
    )
    efectivo = df_metrics['Cash And Cash Equivalents'].fillna(0)
    df_metrics['EnterpriseValue'] = df_metrics['MarketCap'] + deuda_total - efectivo

    # Ratios de valuación
    df_metrics['PE_Trailing'] = df_metrics['MarketCap'] / df_metrics['Net Income']
    df_metrics['EnterpriseToEbitda'] = df_metrics['EnterpriseValue'] / df_metrics['EBITDA']
    
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['PriceToBook'] = df_metrics['MarketCap'] / df_metrics['Stockholders Equity']

    # Ratios de rentabilidad y márgenes
    df_metrics['operatingMargins'] = df_metrics['Operating Income'] / df_metrics['Total Revenue']
    df_metrics['profitMargins'] = df_metrics['Net Income'] / df_metrics['Total Revenue']
    
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['returnOnEquity'] = df_metrics['Net Income'] / df_metrics['Stockholders Equity']
        
    if 'Total Assets' in df_metrics.columns:
        df_metrics['ReturnOnAssets'] = df_metrics['Net Income'] / df_metrics['Total Assets']

    # Ratios de liquidez y solvencia
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['debtToEquity'] = deuda_total / df_metrics['Stockholders Equity']
        
    if 'Current Assets' in df_metrics.columns and 'Current Liabilities' in df_metrics.columns:
        df_metrics['currentRatio'] = df_metrics['Current Assets'] / df_metrics['Current Liabilities']
       

    # Ratios de crecimiento
    # Crecimiento interanual (Year-over-Year, YoY) - Ventana de 12 meses y Trimestral (QoQ)
    # Se aplica .abs() en el denominador para corregir matemáticamente 
    # la dirección del crecimiento si el periodo anterior era negativo.
    if 'Total Revenue' in df_metrics.columns:
        prev_rev_12 = df_metrics.groupby('Ticker')['Total Revenue'].shift(12)
        df_metrics['Revenue_YoY'] = (df_metrics['Total Revenue'] - prev_rev_12) / prev_rev_12.abs()

        prev_rev_3 = df_metrics.groupby('Ticker')['Total Revenue'].shift(3)
        df_metrics['Revenue_QoQ'] = (df_metrics['Total Revenue'] - prev_rev_3) / prev_rev_3.abs()
    
    if 'EBITDA' in df_metrics.columns:
        prev_ebitda_12 = df_metrics.groupby('Ticker')['EBITDA'].shift(12)
        df_metrics['EBITDA_YoY'] = (df_metrics['EBITDA'] - prev_ebitda_12) / prev_ebitda_12.abs()

        prev_ebitda_3 = df_metrics.groupby('Ticker')['EBITDA'].shift(3)
        df_metrics['EBITDA_QoQ'] = (df_metrics['EBITDA'] - prev_ebitda_3) / prev_ebitda_3.abs()
    
    if 'Free Cash Flow' in df_metrics.columns:
        prev_FCF_12 = df_metrics.groupby('Ticker')['Free Cash Flow'].shift(12)
        df_metrics['FCF_YoY'] = (df_metrics['Free Cash Flow'] - prev_FCF_12) / prev_FCF_12.abs()

        prev_FCF_3 = df_metrics.groupby('Ticker')['Free Cash Flow'].shift(3)
        df_metrics['FCF_QoQ'] = (df_metrics['Free Cash Flow'] - prev_FCF_3) / prev_FCF_3.abs()

    if 'Capital Expenditure' in df_metrics.columns:
        prev_Capex_12 = df_metrics.groupby('Ticker')['Capital Expenditure'].shift(12)
        df_metrics['Capex_YoY'] = (df_metrics['Capital Expenditure'] - prev_Capex_12) / prev_Capex_12.abs()

        prev_Capex_3 = df_metrics.groupby('Ticker')['Capital Expenditure'].shift(3)
        df_metrics['Capex_QoQ'] = (df_metrics['Capital Expenditure'] - prev_Capex_3) / prev_Capex_3.abs()

    # Nuevas columnas
    # Apalancamiento
    # Se añade un pequeño epsilon (1e-6) al denominador para evitar divisiones por cero
    df_metrics['NetDebt_to_EBITDA'] = (deuda_total - df_metrics['Cash And Cash Equivalents']) / (df_metrics['EBITDA'] + 1e-6)

    # Free Cash Flow Conversion
    df_metrics['FCF_to_EBITDA'] = df_metrics['Free Cash Flow'] / (df_metrics['EBITDA'] + 1e-6)

    # Capital Intensity (Intensidad de capital)
    # Usamos np.abs porque el Capex suele reportarse en negativo en los estados de flujo de caja
    df_metrics['Capex_to_Revenue'] = np.abs(df_metrics['Capital Expenditure']) / (df_metrics['Total Revenue'] + 1e-6)
    
    # Limpieza de posibles infinitos creados por EBITDAs muy cercanos a cero
    cols_nuevas = ['NetDebt_to_EBITDA', 'FCF_to_EBITDA', 'Capex_to_Revenue']
    df_metrics[cols_nuevas] = df_metrics[cols_nuevas].replace([np.inf, -np.inf], np.nan)

    # Limpieza final
    # Redondear para legibilidad y consistencia
    cols_a_redondear = [
        'PE_Trailing', 'PriceToBook', 'EnterpriseToEbitda', 'operatingMargins', 
        'profitMargins', 'returnOnEquity', 'ReturnOnAssets', 'debtToEquity', 'currentRatio'
    ]
    
    # Aplicar redondeo solo a las columnas que realmente existen en el df_metrics
    cols_presentes = [col for col in cols_a_redondear if col in df_metrics.columns]
    df_metrics[cols_presentes] = df_metrics[cols_presentes].round(4) # 4 decimales para capturar bien los porcentajes

    return df_metrics


def calcular_retornos(df_precios: pd.DataFrame, df_index: pd.DataFrame, ventana: int = 12, min_periodos: int = None) -> pd.DataFrame:
    """
    Calcula retornos mensuales, varianza del activo y covarianza con el mercado.
    """
    if min_periodos is None:
        min_periodos = int(ventana * 0.75) # establece ventana minima para los primeros valores
        
    ticker_mercado = df_index['Ticker'].iloc[0]
    
    # Preparar datos y calcular retornos
    df_unido = pd.concat([df_precios, df_index], ignore_index=True)
    df_pivot = df_unido.pivot(index='Fecha', columns='Ticker', values='Close').sort_index()
    df_retornos = df_pivot.pct_change(fill_method=None).dropna(how='all')
    
    retornos_mercado = df_retornos[ticker_mercado]
    df_activos = df_retornos.drop(columns=[ticker_mercado])
    
    # Calcular estadísticas móviles (Features)
    varianzas_activos = df_activos.rolling(window=ventana, min_periods=min_periodos).var()
    covarianzas = df_activos.rolling(window=ventana, min_periods=min_periodos).cov(retornos_mercado)
    
    # Transformar cada matriz a formato largo (melt)
    df_ret_long = df_activos.reset_index().melt(
        id_vars='Fecha', var_name='Ticker', value_name='Retorno_Mensual'
    )
    df_var_long = varianzas_activos.reset_index().melt(
        id_vars='Fecha', var_name='Ticker', value_name='Varianza_Activo'
    )
    df_cov_long = covarianzas.reset_index().melt(
        id_vars='Fecha', var_name='Ticker', value_name='Covarianza_Mercado'
    )
    
    # Consolidar todas las features en un solo DataFrame
    df_features = df_ret_long.merge(df_var_long, on=['Fecha', 'Ticker'])
    df_features = df_features.merge(df_cov_long, on=['Fecha', 'Ticker'])
    
    # Limpiar registros sin suficientes datos (NaNs de la ventana inicial)
    df_features = df_features.dropna(subset=['Varianza_Activo', 'Covarianza_Mercado']).reset_index(drop=True)
    
    # Redondear para mayor legibilidad
    df_features = df_features.round(6)
    
    return df_features


# Funciones "legacy": ya no se utilizan en el código actual, las dejo por las dudas.

'''
from fredapi import Fred
from src.data_sources import fred_api_key
# Extraer datos macroeconómicos de FRED (opcional, si se quiere enriquecer el dataset con indicadores macro)
def extraer_datos_macro(indicadores: list) -> pd.DataFrame:
    """
    Extrae datos macroeconómicos de FRED para una lista de indicadores y un rango de fechas.
    """
    if not fred_api_key:
        print("Advertencia: No se ha proporcionado una clave API de FRED. No se podrán extraer datos macroeconómicos.")
        return pd.DataFrame()
    else:
        # Fecha Inicial: fecha actual menos el periodo definido (ej. 4 años atrás)
        fecha_final = datetime.now()
        if periodo.endswith('y'):
            anios = int(periodo[:-1])
        else:
            anios = 4

        fecha_inicial = fecha_final.replace(year=fecha_final.year - anios)

        fred = Fred(api_key=fred_api_key)
        dfs_macro = []
        for indicador in indicadores:
            try:
                df_indicador = fred.get_series(indicador, observation_start=fecha_inicial)
                df_indicador = df_indicador.reset_index()
                df_indicador.columns = ['Fecha', indicador]
                dfs_macro.append(df_indicador)
            except Exception as e:
                print(f"Error extrayendo {indicador} de FRED: {e}")
                continue
        if dfs_macro:
            # Establecer Fecha como índice para cada dataframe
            for df in dfs_macro:
                df.set_index('Fecha', inplace=True)
            
            # Concatenar en axis=1 (columnas)
            df_macro = pd.concat(dfs_macro, axis=1)
            
            # Eliminar columnas duplicadas (si las hay)
            df_macro = df_macro.loc[:,~df_macro.columns.duplicated()]
            
            # Convertir a frecuencia mensual (último valor del mes)
            df_macro = df_macro.resample('ME').last()
            
            # Eliminar filas completamente vacías
            df_macro = df_macro.dropna(how='all')
            
            # Resetear índice para tener Fecha como columna
            df_macro = df_macro.reset_index()
            df_macro.columns.name = None
            
            return df_macro
        else:
            return pd.DataFrame()


def calcular_growth_features(df:pd.DataFrame, cols:list)->pd.DataFrame:
    for col in cols:
        try:
            df[f'{col}_QoQ'] = df[col].pct_change(3, fill_method=None)
            df[f'{col}_YoY'] = df[col].pct_change(12, fill_method=None)
            
        except Exception as e:
            print(f"Error procesando la columna {col}: {e}")
            continue
    
    return df



def extraer_info(tickers_list:list)->pd.DataFrame:
    """Extrae última información fundamental, sin datos historicos."""
    dfs_info = []

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info   
    
            # Validación
            if not isinstance(info, dict) or len(info) == 0:
                print(f"Sin datos para {ticker}")
                continue
    
            # Seleccionar campos
            row = {
                'Ticker': ticker,
                'Sector': info.get('sector'),
                'MarketCap': info.get('marketCap'),
                'Beta': info.get('beta'),
                'DividendYield': info.get('dividendYield'),
                'ForwardPE': info.get('forwardPE'),
                'trailingPegRatio': info.get('trailingPegRatio'),
                'PriceToBook': info.get('priceToBook'),
                'EnterpriseToEbitda': info.get('enterpriseToEbitda'),
                'ReturnOnAssets': info.get('returnOnAssets'),
                'returnOnEquity': info.get('returnOnEquity'),
                'profitMargins': info.get('profitMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'currentRatio': info.get('currentRatio'),
                'debtToEquity': info.get('debtToEquity'),
                'revenueGrowth': info.get('revenueGrowth'),
                'shortPercentOfFloat': info.get('shortPercentOfFloat')
            }
    
            dfs_info.append(row)
    
        except Exception as e:
            print(f"Error con {ticker}: {e}")
            continue
    
    if dfs_info:
        return pd.DataFrame(dfs_info)
    else:
        return pd.DataFrame()
'''

# Bloque principal para pruebas desde el terminal
# ejecutar desde la raiz: python -m src.ingestion

def main():
    # Prueba de extraer_datos_macro()
    # indicadores_prueba = ['FEDFUNDS', 'GS10', 'T10Y2Y', 'CPIAUCSL', 'UNRATE']
    # df_macro = extraer_datos_macro(indicadores_prueba)
    # print("Datos macroeconómicos extraídos de FRED:")
    # print(df_macro)

    # Prueba de extraer_financials()
    tickers_prueba = ["MSFT", "NVDA"]
    df_fundamentals = extraer_financials(tickers_prueba)
    df_fundamentals.to_csv("data/fundamentals_test.csv")

if __name__ == "__main__":
    main()