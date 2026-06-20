"""
src/extract.py
Módulo de la fase de Ingestión de Datos
"""
import pandas as pd
import numpy as np
import urllib.request
from datetime import datetime
import os
import yfinance as yf
import simfin as sf
from src.data_sources import simfin_api_key
from src.config import (
    periodo, intervalo, cols_resultados, cols_balance,
    cols_cashflow, data_folder, mapa_columnas,
    retardo_publicacion
)
 


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

def clean_ticker(s):
    """
    Función auxiliar para la limpieza de tickers
    """
    if pd.isna(s):
        return None # para evitar crear ticker 'nan'

    return (
        str(s)
        .strip() # Eliminar espacios al principio y al final
        .replace('.', '-')   # BRK.B -> BRK-B
        .replace(' ', '') # Eliminar espacios intermedios
    )


def limpieza_tickers(df:pd.DataFrame)->pd.DataFrame:
    # Seleccionar y renombrar columnas
    df_tickers = df[["Symbol", "GICS Sector", "Date added"]].copy()
    df_tickers.rename(columns={
        "Symbol": "Ticker",
        "GICS Sector": "Sector",
        "Date added": "DateAdded"
        }, inplace=True)
    
    # Limpieza de tickers
    df_tickers['Ticker'] = df_tickers['Ticker'].map(clean_ticker)

    # Eliminar espacios intermedios en los nombres de los sectores
    df_tickers["Sector"] = df_tickers["Sector"].str.replace(" ", "")   

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
    df_prices.drop(['High', 'Low', 'Volume', 'Capital Gains', 'Stock Splits'], axis=1, inplace=True, errors='ignore')

    # Quitar la zona horaria
    df_prices['Date'] = pd.to_datetime(df_prices['Date']).dt.tz_localize(None)

    return df_prices


def alinear_fecha_trimestral(fecha):
    """
    Desplaza una fecha hacia el próximo 1er día de los meses 3, 6, 9 o 12.
    Postura estricta: Si la publicación es el mismo día 1, se pasa al trimestre
    siguiente para evitar lookahead bias en precios de apertura.
    """
    if pd.isna(fecha):
        return fecha
        
    m = fecha.month
    y = fecha.year
    
    # Al usar estrictamente '<', cualquier publicación en marzo (incluyendo el día 1) 
    # salta automáticamente a junio.
    # Aún si la publicación hubiese sido el mismo primer día del trimestre antes de la apertura (BMO),
    # cualquier gap en el precio de apertura habría sido inalcanzable. 
    if m < 3:
        return pd.Timestamp(year=y, month=3, day=1)
    elif m < 6:
        return pd.Timestamp(year=y, month=6, day=1)
    elif m < 9:
        return pd.Timestamp(year=y, month=9, day=1)
    elif m < 12:
        return pd.Timestamp(year=y, month=12, day=1)
    else:
        # Diciembre (mes 12) completo pasa a marzo del año siguiente
        return pd.Timestamp(year=y + 1, month=3, day=1)
    

def extraer_financials(tickers_list: list, aproximar_fechas: bool = False) -> pd.DataFrame:
    """
    Extrae datos financieros trimestrales del Estado de Resultados, Balance General y Cash Flow.
    
    Parámetros:
    - tickers_list: Lista de símbolos a extraer.
    - aproximar_fechas: Si es True, usa una estimación estática de días (más rápido). 
                        Si es False, busca las fechas reales de publicación (más lento, evita lookahead bias).
    """
    dfs_lista = []       

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Se extraen los Estados trimestrales
            fin = yf_ticker.quarterly_financials
            bal = yf_ticker.quarterly_balance_sheet
            cf = yf_ticker.quarterly_cashflow

            # Validación del Estado de Resultados
            if fin is None or fin.empty:
                print(f"Sin datos financieros trimestrales para {ticker}")
                continue

            # Se limitan a las primeras 4 columnas 
            fin = fin.iloc[:, :4]

            # Transponer y filtrar Estado de Resultados
            df_fin = fin.T.reindex(columns=cols_resultados)

            # Validación del Balance General (si existe)
            if bal is not None and not bal.empty:
                bal = bal.iloc[:, :4] 
                df_bal = bal.T.reindex(columns=cols_balance)
                df_temp = df_fin.join(df_bal, how='left')
            else:
                df_temp = df_fin.copy() 
                for col in cols_balance:
                    df_temp[col] = float('nan') 
                df_temp[cols_balance] = df_temp[cols_balance].astype(float)  

            # Validación del Cash Flow (si existe)
            if cf is not None and not cf.empty:
                cf = cf.iloc[:, :4] 
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

            # --- TRANSFORMACIÓN DE FECHAS #1: PUBLICACIÓN ---
            fechas_datetime = pd.to_datetime(df_temp['Date']).dt.tz_localize(None)

            if aproximar_fechas:
                # MODO RÁPIDO: Saltea la API de earnings y aplica vectorización directa
                df_temp['Date'] = (fechas_datetime + pd.Timedelta(days=retardo_publicacion)).dt.normalize()
            else:
                # MODO ESTRICTO: Petición a la API para fechas reales
                try:
                    df_earnings = yf_ticker.earnings_dates
                    if df_earnings is not None and not df_earnings.empty:
                        pub_dates = df_earnings.index.tz_localize(None).sort_values()

                        real_dates = []
                        for q_end in fechas_datetime:
                            valid_pubs = pub_dates[
                                (pub_dates > q_end)
                                & (pub_dates <= q_end + pd.Timedelta(days=90))
                            ]
                            if not valid_pubs.empty:
                                real_dates.append(valid_pubs[0].normalize())
                            else:
                                real_dates.append((q_end + pd.Timedelta(days=retardo_publicacion)).normalize())
                        df_temp['Date'] = real_dates
                    else:
                        df_temp['Date'] = (fechas_datetime + pd.Timedelta(days=retardo_publicacion)).dt.normalize()
                except Exception as e:
                    print(f"Aviso: Error obteniendo fechas reales para {ticker}. Usando estimación. ({e})")
                    df_temp['Date'] = (fechas_datetime + pd.Timedelta(days=retardo_publicacion)).dt.normalize()

            # --- TRANSFORMACIÓN DE FECHAS #2: ALINEACIÓN TRIMESTRAL ---
            df_temp['Date'] = pd.to_datetime(df_temp['Date']).apply(alinear_fecha_trimestral)

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


def extraer_simfin(tickers_validos: list) -> pd.DataFrame:
    sf.set_api_key(simfin_api_key)
    sf.set_data_dir(f'{data_folder}/simfin')

    # Identificar columnas de metadatos compartidas
    metadatos_comunes = [
        'SimFinId', 'Currency', 'Fiscal Year', 'Fiscal Period', 
        'Publish Date', 'Restated Date', 'Shares (Basic)', 'Shares (Diluted)',
        'Depreciation & Amortization', 'Provision for Loan Losses'
    ]

    # Función auxiliar para procesar cualquier sector
    def procesar_sector(func_inc, func_bal, func_cf) -> pd.DataFrame:
        """Descarga, limpia y une los reportes de un sector específico."""
        df_inc = func_inc(variant='quarterly')
        df_bal = func_bal(variant='quarterly')
        df_cf = func_cf(variant='quarterly')

        # Eliminar metadatos verificando las columnas del DF actual
        cols_drop_bal = [col for col in metadatos_comunes if col in df_bal.columns]
        cols_drop_cf = [col for col in metadatos_comunes if col in df_cf.columns]
        
        df_bal_clean = df_bal.drop(columns=cols_drop_bal)
        df_cf_clean = df_cf.drop(columns=cols_drop_cf)

        # Unión usando el índice nativo ('Ticker' y 'Report Date')
        return df_inc.join(df_bal_clean, how='outer').join(df_cf_clean, how='outer')

    # Obtener los DataFrames de los 3 sectores
    df_general = procesar_sector(sf.load_income, sf.load_balance, sf.load_cashflow)
    df_banks = procesar_sector(sf.load_income_banks, sf.load_balance_banks, sf.load_cashflow_banks)
    df_insurance = procesar_sector(sf.load_income_insurance, sf.load_balance_insurance, sf.load_cashflow_insurance)

    # Concatenar 
    # Al tener distinto esquema de columnas, Pandas llenará con NaN donde no aplique.
    df_consolidado = pd.concat([df_general, df_banks, df_insurance], axis=0)

    # Resetear el índice
    df_consolidado = df_consolidado.reset_index()

    # Limpieza de tickers usando la misma función auxiliar que los Tickers S&P 500
    df_consolidado['Ticker'] = df_consolidado['Ticker'].map(clean_ticker)

    # Se seleccionan los tickers del S&P500
    #df_final = df_consolidado[df_consolidado['Ticker'].isin(tickers_validos)]
    df_final = df_consolidado.copy()
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

    # Llevar la fecha de publicación real al primer día del trimestre siguiente
    df['Date'] = pd.to_datetime(df['Publish Date']).apply(alinear_fecha_trimestral)

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


def limpieza_final(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar posibles fechas futuras
    # Se hace antes del ffill para no propagar datos hacia trimestres irreales
    hoy = pd.Timestamp.today().normalize()
    df = df[df['Date'] <= hoy]

    # Ordenar cronológicamente para el Forward Fill
    df = df.sort_values(by=['Ticker', 'Date'])

    # Aplicar Forward Fill a las columnas financieras 
    # con limite de 3, no deben haber huecos para calcular los ratios TTM (trailing twelve months)
    # (Asume que cols_resultados, cols_balance y cols_cashflow están definidas globalmente o pasadas como argumento)
    cols_financieras = cols_resultados + cols_balance + cols_cashflow
    df[cols_financieras] = df.groupby('Ticker')[cols_financieras].ffill(limit=3)

    # Eliminar las filas anteriores al primer reporte financiero disponible
    columna_critica = 'EBITDA' # es necesaria para los ratios
    df_clean = df.dropna(subset=[columna_critica])

    # Resetear el índice para que quede de 0 a N
    df_clean = df_clean.reset_index(drop=True)

    # Eliminar espacios en los nombres de las columnas
    df_clean.columns = df_clean.columns.str.replace(' ', '')

    return df_clean


# Función auxiliar para analizar el retardo de publicación


def analizar_retardo_publicacion(tickers_list: list) -> pd.Series:
    """
    Calcula los días de retardo promedio entre el cierre del trimestre y la 
    fecha real de publicación de resultados para una lista de tickers.
    
    Devuelve una Serie de Pandas con todos los retardos individuales encontrados.
    """
    lista_retardos = []

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # 1. Obtener cierres de trimestre (las columnas del Estado de Resultados)
            fin = yf_ticker.quarterly_financials
            if fin is None or fin.empty:
                continue
            
            # Convertir a datetime sin zona horaria
            fechas_cierre = pd.to_datetime(fin.columns).tz_localize(None)
            
            # 2. Obtener fechas de publicación históricas
            df_earnings = yf_ticker.earnings_dates
            if df_earnings is None or df_earnings.empty:
                continue
                
            pub_dates = df_earnings.index.tz_localize(None).sort_values()
            
            # 3. Calcular la diferencia para cada trimestre
            for q_end in fechas_cierre:
                # Buscamos la primera publicación posterior al cierre (máximo 90 días)
                valid_pubs = pub_dates[(pub_dates > q_end) & (pub_dates <= q_end + pd.Timedelta(days=90))]
                
                if not valid_pubs.empty:
                    # Guardamos la diferencia exacta en días
                    dias_retardo = (valid_pubs[0] - q_end).days
                    lista_retardos.append(dias_retardo)
                    
        except Exception as e:
            print(f"No se pudo procesar el retardo para {ticker}: {e}")
            continue

    # 4. Procesar y mostrar resultados estadísticos
    if lista_retardos:
        series_retardos = pd.Series(lista_retardos)
        
        print("\n" + "="*40)
        print("  ESTADÍSTICAS DEL RETARDO DE PUBLICACIÓN")
        print("="*40)
        print(f"Total de trimestres analizados : {len(series_retardos)}")
        print(f"Promedio de retardo (días)    : {series_retardos.mean():.2f}")
        print(f"Mediana de retardo (días)     : {series_retardos.median():.1f}")
        print(f"Mínimo retardo registrado     : {series_retardos.min()} días")
        print(f"Máximo retardo registrado     : {series_retardos.max()} dias")
        print(f"Percentil 75 (75% reporta <)  : {series_retardos.quantile(0.75):.1f} días")
        print(f"Percentil 90 (90% reporta <)  : {series_retardos.quantile(0.90):.1f} días")
        print("="*40 + "\n")
        
        return series_retardos
    else:
        print("No se encontraron suficientes datos históricos para calcular los retardos.")
        return pd.Series()


"""
Se obtuvo el siguiente resultado del análisis efectuado sobre el universo de tickers:

========================================
  ESTADÍSTICAS DEL RETARDO DE PUBLICACIÓN
========================================
Total de trimestres analizados : 2353
Promedio de retardo (días)    : 30.32
Mediana de retardo (días)     : 30.0
Mínimo retardo registrado     : 3 días
Máximo retardo registrado     : 61 dias
Percentil 75 (75% reporta <)  : 36.0 días
Percentil 90 (90% reporta <)  : 41.0 días
========================================

Se adopta por defecto un retardo de 30 días (mediana)
Puedes modificar el parámetro en el fichero src/config.py
"""


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

    # Análisis del tiempo medio de retardo en la fecha de publicación
    #tickers_prueba = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'V', 'PG', 'XOM']
    #datos_retardo = analizar_retardo_publicacion(tickers_prueba)

if __name__ == "__main__":
    main()