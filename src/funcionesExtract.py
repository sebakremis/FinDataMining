"""
src/funcionesExtract.py
Módulo de funciones para la fase de Ingestión de Datos (Extract).
"""
import pandas as pd
import urllib.request
from datetime import datetime
import os
import yfinance as yf
from src.config import periodo, intervalo, cols_resultados, cols_balance, cols_cashflow, data_folder, mapa_columnas


def descargar_constituents(force_update=False):
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

def limpieza_tickers(df:pd.DataFrame)->pd.DataFrame:
    # Seleccionar y renombrar columnas
    df_tickers = df[["Symbol", "GICS Sector", "Date added"]].copy()
    df_tickers.rename(columns={
        "Symbol": "Ticker",
        "GICS Sector": "Sector",
        "Date added": "DateAdded"
        }, inplace=True)
    
    # Modificar "BRK.B" a "BRK-B" y "BF.B" a "BF-B" para evitar problemas con yfinance
    df_tickers["Ticker"] = df_tickers["Ticker"].replace("BRK.B", "BRK-B")
    df_tickers["Ticker"] = df_tickers["Ticker"].replace("BF.B", "BF-B")

    # Eliminar espacios en los nombres de los sectores
    df_tickers["Sector"] = df_tickers["Sector"].str.replace(" ", "")

    # Asegurar que los Tickers no tengan espacios en blanco
    df_tickers['Ticker'] = df_tickers['Ticker'].astype(str).str.strip()

    return df_tickers


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
    
    # Eliminar columnas innecesarias (usar errors='ignore' por si no hay splits o Capital Gains)
    df_prices.drop(['Open', 'High', 'Low', 'Volume', 'Capital Gains', 'Stock Splits'], axis=1, inplace=True, errors='ignore')

    # Quitar la zona horaria
    df_prices['Date'] = pd.to_datetime(df_prices['Date']).dt.tz_localize(None)

    return df_prices


def extraer_financials(tickers_list: list) -> pd.DataFrame:
    """
    Extrae datos financieros trimestrales del Estado de Resultados, Balance General y Cash Flow.
    Devuelve un DataFrame unificado para cálculo de ratios históricos limitados a los últimos 4 trimestres.
    """
    dfs_lista = []       

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # 1. Cambiamos a los métodos trimestrales (quarterly_*)
            fin = yf_ticker.quarterly_financials
            bal = yf_ticker.quarterly_balance_sheet
            cf = yf_ticker.quarterly_cashflow

            # Validación del Estado de Resultados
            if fin is None or fin.empty:
                print(f"Sin datos financieros trimestrales para {ticker}")
                continue

            # 2. Limitamos a las primeras 4 columnas (yfinance ordena de más reciente a más antiguo)
            fin = fin.iloc[:, :4]

            # Transponer y filtrar Estado de Resultados
            df_fin = fin.T.reindex(columns=cols_resultados)

            # Validación del Balance General (si existe)
            if bal is not None and not bal.empty:
                bal = bal.iloc[:, :4] # Limitar a 4 trimestres
                df_bal = bal.T.reindex(columns=cols_balance)
                df_temp = df_fin.join(df_bal, how='left')
            else:
                df_temp = df_fin.copy() 
                for col in cols_balance:
                    df_temp[col] = float('nan') 
                df_temp[cols_balance] = df_temp[cols_balance].astype(float)  

            # Validación del Cash Flow (si existe)
            if cf is not None and not cf.empty:
                cf = cf.iloc[:, :4] # Limitar a 4 trimestres
                df_cf = cf.T.reindex(columns=cols_cashflow)
                df_temp = df_temp.join(df_cf, how='left')
            else:
                for col in cols_cashflow:
                    df_temp[col] = float('nan')
                df_temp[cols_cashflow] = df_temp[cols_cashflow].astype(float)        

            # Limpieza del DataFrame temporal
            df_temp = df_temp.reset_index()
            df_temp = df_temp.rename(columns={'index': 'Date'})
            df_temp['Ticker'] = ticker 

            # --- TRANSFORMACIÓN DE FECHAS ---
            # Para evitar "Lookahead Bias", asumimos que la información fue pública 30 días después del cierre.
            fechas_datetime = pd.to_datetime(df_temp['Date']).dt.tz_localize(None)

            df_temp['Date'] = (fechas_datetime + pd.Timedelta(days=30)).dt.normalize()

            # Se convierte al primer dia del mes siguiente para alinear con precios mensuales
            df_temp['Date'] = (df_temp['Date'] + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)

            if not df_temp.empty:
                dfs_lista.append(df_temp)

        except Exception as e:
            print(f"Error procesando fundamentales para {ticker}: {e}")
            continue

    # Concatenación final
    if dfs_lista:
        df_final = pd.concat(dfs_lista, axis=0, ignore_index=True)
        return df_final
    else:
        return pd.DataFrame()


def limpieza_final(df:pd.DataFrame)->pd.DataFrame:
    # Ordenar cronológicamente para el Forward Fill
    df = df.sort_values(by=['Ticker', 'Date'])

    # Aplicar Forward Fill a las columnas financieras
    cols_financieras = cols_resultados + cols_balance + cols_cashflow

    df[cols_financieras] = df.groupby('Ticker')[cols_financieras].ffill()

    # Eliminar las filas anteriores al primer reporte financiero disponible
    columna_critica = 'EBITDA' # es necesaria para los ratios
    df_clean = df.dropna(subset=[columna_critica])

    # Resetear el índice para que quede de 0 a N
    df_clean = df_clean.reset_index(drop=True)

    # Eliminar espacios en los nombres de las columnas
    df_clean.columns = df_clean.columns.str.replace(' ', '')

    return df_clean



import simfin as sf
from src.data_sources import simfin_api_key
def extraer_simfin(tickers_validos:list) -> pd.DataFrame:
    sf.set_api_key(simfin_api_key)
    sf.set_data_dir(data_folder)

    # Cargar reportes trimestrales
    df_fin = sf.load_income(variant='quarterly')
    df_bal = sf.load_balance(variant='quarterly')
    df_cf = sf.load_cashflow(variant='quarterly')

    # Identificar columnas de metadatos compartidas que generarían sufijos (_x, _y)
    # Se mantienen estas columnas SOLO en el dataframe de Resultados (df_fin)
    metadatos_comunes = [
        'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period', 
        'Publish Date', 'Restated Date', 'Shares (Basic)', 'Shares (Diluted)',
        'Depreciation & Amortization'
    ]

    # 3. Eliminar metadatos de Balance y Cash Flow antes de unir
    cols_drop_bal = [col for col in metadatos_comunes if col in df_bal.columns]
    cols_drop_cf = [col for col in metadatos_comunes if col in df_cf.columns]
    
    df_bal_clean = df_bal.drop(columns=cols_drop_bal)
    df_cf_clean = df_cf.drop(columns=cols_drop_cf)

    # Unión usando el índice nativo ('Ticker' y 'Report Date')
    # Se usa how='outer' para no perder la fila si una empresa publicó Income pero se retrasó en Cash Flow
    df_simfin = df_fin.join(df_bal_clean, how='outer').join(df_cf_clean, how='outer')

    # Resetear el índice para que Ticker y Report Date vuelvan a ser columnas normales
    df_simfin = df_simfin.reset_index()

    # Se seleccionan los tickers del S&P 500
    df_final = df_simfin[df_simfin['Ticker'].isin(tickers_validos)]

    return df_final

def estandarizar_simfin(df_raw:pd.DataFrame, cols:list) -> pd.DataFrame:
    df = df_raw.copy()
    df.rename(columns=mapa_columnas, inplace=True)

    # Calcular las columnas faltantes
    df['EBITDA'] = df['Operating Income'] - df['Depreciation & Amortization'] # se resta por ser valores negativos
    df['Total Debt'] = df['Current Debt'] + df['Long Term Debt']
    df['Free Cash Flow'] = df['Operating Cash Flow'] + df['Capital Expenditure'] # se suman por ser negativos

    # Asegurar que Publish Date es un formato datetime
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])

    # Llevar la fecha de publicación real al primer día del mes siguiente (para unir con precios mensuales)
    df['Date'] = (df['Publish Date'] + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)

    #  Quitar las columnas que ya no hacen falta para que coincida con yfinance
    df = df[cols]

    return df


def unir_financials(df_yfinance:pd.DataFrame, df_simfin:pd.DataFrame)->pd.DataFrame:
    # Eliminar posibles duplicados INTERNOS en cada DataFrame antes de cruzarlos
    # Se mantiene el último como válido (dato reexpresado/corregido)   
    df_yf_unique = df_yfinance.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
    df_sf_unique = df_simfin.drop_duplicates(subset=['Ticker', 'Date'], keep='last') 

    # Convertir 'Ticker' y 'Date' en el índice de ambos DataFrames temporalmente
    df_sf_idx = df_sf_unique.set_index(['Ticker', 'Date'])
    df_yf_idx = df_yf_unique.set_index(['Ticker', 'Date'])

    # Aplicar combine_first para tratar el solapamiento
    # Toma el valor de yfinance, si fuese NaN lo intenta rellenar con SimFin
    df_financials_idx = df_yf_idx.combine_first(df_sf_idx)

    # Se devuelven 'Ticker' y 'Date' como columnas normales
    df_unido = df_financials_idx.reset_index()

    # Auditoría de solapamientos
    mascara_duplicados = df_unido.duplicated(subset=['Ticker', 'Date'], keep=False)
    df_solapado = df_unido[mascara_duplicados]
    print(f"Se han encontrado {len(df_solapado)} filas con Ticker y Date solapados.")

    return df_unido



# Funciones "legacy": ya no se utilizan en el código actual, las dejo por las dudas.

'''
from fredapi import Fred
from src.data_sources import fred_api_key
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
                df_indicador.columns = ['Date', indicador]
                dfs_macro.append(df_indicador)
            except Exception as e:
                print(f"Error extrayendo {indicador} de FRED: {e}")
                continue
        if dfs_macro:
            # Establecer Fecha como índice para cada dataframe
            for df in dfs_macro:
                df.set_index('Date', inplace=True)
            
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
# ejecutar desde la raiz: python -m src.funcionesExtract

def main():
    # Prueba de extraer_financials()
    #tickers_prueba = ["CVNA"]
    #df_fundamentals = extraer_financials(tickers_prueba)
    #df_fundamentals.to_csv(f"{data_folder}/fundamentals_test.csv")

    # Prueba de extraer_simfin()
    df = extraer_simfin()
    print(df.head())

if __name__ == "__main__":
    main()