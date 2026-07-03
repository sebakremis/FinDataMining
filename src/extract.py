"""
src/extract.py
Módulo de la fase de Ingestión de Datos
"""
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
from datetime import datetime
import os
import yfinance as yf
import simfin as sf
import time
import random
from src.data_sources import simfin_api_key
from src.config import (
    periodo, intervalo, cols_resultados, cols_balance,
    cols_cashflow, data_folder, mapa_columnas,
    retardo_publicacion, cambios_tickers,
    ruta_sin_datos, raw_data_file
)

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


def extraer_simfin() -> pd.DataFrame:
    """
    Descarga todos los datos de simFin, sin filtrar tickers.
    """
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
    
    return df_consolidado


def generar_universo_tickers(df_simfin_raw: pd.DataFrame, 
                             umbral_filas: int = 19, 
                             columna_ranking: str = 'Revenue', 
                             cantidad_tickers: int = 550)->list:
    """
    Se genera el fichero del universo de tickers obtendidos de finSim:
    1. Filtra los tickers que superen el umbral mínimo de trimestres disponibles.
    2. Agrupa por ticker y calcula el promedio de su métrica para ranquearlos.
    3. Selecciona el top N y devuelve la lista.
    """
    df = df_simfin_raw.copy()
    # Normalizar tickers: limpieza y corrección de nombres 
    df['Ticker'] = df['Ticker'].map(clean_ticker)
    df['Ticker'] = df['Ticker'].map(lambda t: cambios_tickers.get(t, t))


    # Filtrar tickers sin datos
    if ruta_sin_datos.exists():
        df_excluidos = pd.read_csv(ruta_sin_datos)
        
        if 'Ticker' in df_excluidos.columns:
            df_excluidos['Ticker'] = df_excluidos['Ticker'].map(clean_ticker)
            tickers_excluidos = df_excluidos['Ticker'].tolist()
            df = df[~df['Ticker'].isin(tickers_excluidos)]

    # Contar y filtrar tickers válidos
    counts = df.groupby("Ticker").size()
    tickers_validos = counts[counts >= umbral_filas].index
    df_filtrado = df[df["Ticker"].isin(tickers_validos)]
    
    # Calcular un valor representativo de 'Revenue' por empresa.
    ranking_metricas = df_filtrado.groupby("Ticker")[columna_ranking].mean()
    
    # Se ordenan de mayor a menor y tomar el Top N
    top_tickers = ranking_metricas.sort_values(ascending=False).head(cantidad_tickers).index.tolist()

    # Asegurar que no existan duplicados preservando el orden
    tickers_unicos = list(dict.fromkeys([str(t) for t in top_tickers]))
    
    return tickers_unicos


def estandarizar_simfin(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    
    # Renombrar las columnas según el mapa
    df.rename(columns=mapa_columnas, inplace=True)

    # Se crean las columnas faltantes vacías, luego serán calculadas en la fase de transformación
    cols_faltantes = ['EBITDA', 'Total Debt', 'Free Cash Flow']
    df[cols_faltantes] = np.nan

    # Salvavidas para evitar NaT masivos
    # simFin deja 'Report Date' tras resetear el índice en extraer_simfin
    if 'Report Date' in df.columns:
        if 'Publish Date' in df.columns:
            # Rellenar vacíos de Publish Date con Report Date
            df['Publish Date'] = df['Publish Date'].fillna(df['Report Date'])
        else:
            # Si Publish Date no existe, usar Report Date directamente
            df['Publish Date'] = df['Report Date']

    # Asegurar que Publish Date es un formato datetime
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])

    # Llevar la fecha de publicación real al primer día del mes siguiente
    df['Date'] = df['Publish Date'] + pd.offsets.MonthBegin(1)

    #  Extraer los nombres de las columnas destino de yfinance desde el diccionario
    cols_yfinance = list(mapa_columnas.values())
    
    # Combinar las columnas del mapa con las creadas manualmente.
    # Se usa dict.fromkeys() para eliminar posibles duplicados manteniendo el orden.
    columnas_deseadas = list(dict.fromkeys(cols_yfinance + cols_faltantes + ['Date', 'Ticker']))
    
    # Filtrar columnas que realmente existan en el df actual
    columnas_finales = [col for col in columnas_deseadas if col in df.columns]

    # Quitar las columnas que ya no hacen falta
    df = df[columnas_finales]

    # Se agrega columna 'FinancialsSource' para indicar que los datos provienen de simFin
    df['FinancialsSource'] = 'simFin'

    # Eliminar posibles fechas futuras generadas por la alineación mensual
    hoy = pd.Timestamp.today().normalize()
    df = df[df['Date'] <= hoy]

    return df


def gestionar_actualizacion(tickers_universe_list: list, df_simfin_standarized:pd.DataFrame) -> tuple[list, pd.DataFrame]:

    
    # Si el archivo no existe, devuelve la lista completa
    if not raw_data_file.exists():
        return tickers_universe_list, df_simfin_standarized
    
    # Leer del fichero guardado las columnas necesarias para la validación
    df_existente = pd.read_parquet(raw_data_file, columns=['Ticker', 'Date'])
    
    # Asegurar consistencia de tipos de datos antes del cruce
    df_existente['Date'] = pd.to_datetime(df_existente['Date'])

    # Actualizar simFin
    # Se filtran las filas de df_simfin_standarized cuyos Ticker y Date no esten en el df_existente
    # Realizamos un 'left join' indicando de dónde proviene cada fila
    df_merged = df_simfin_standarized.merge(
        df_existente[['Ticker', 'Date']].drop_duplicates(), 
        on=['Ticker', 'Date'], 
        how='left', 
        indicator=True
    )
    
    # Filtramos las filas que solo existen en el DataFrame izquierdo (df_simfin_standarized)
    # y luego eliminamos la columna auxiliar '_merge'
    df_simfin_new_data = df_merged[df_merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Actualizar yfinance
    hoy = pd.Timestamp.today().normalize()
    fecha_ultimo_precio = hoy.replace(day=1)

    # Verificar si los tickers están actualizados en el fichero
    # Agrupar por Ticker para obtener la fecha más reciente almacenada para cada uno
    ultimas_fechas_guardadas = df_existente.groupby('Ticker')['Date'].max()

    new_prices_tickers = []
    
    # Verificar si los tickers están actualizados en el fichero
    for ticker in tickers_universe_list:
        # Condición 1: El ticker ni siquiera existe en el archivo parquet
        if ticker not in ultimas_fechas_guardadas:
            new_prices_tickers.append(ticker)
        # Condición 2: El ticker existe, pero su última fecha es anterior al último cierre requerido
        elif ultimas_fechas_guardadas[ticker] < fecha_ultimo_precio:
            new_prices_tickers.append(ticker)

 
    return new_prices_tickers, df_simfin_new_data


def extraer_precios(tickers_list: list) -> tuple[pd.DataFrame,list]:
    """
    Extrae precios históricos en lote y formatea las fechas para cruzar con datos fundamentales.
    Retorna un DataFrame con los precios y una lista con los tickers que no arrojaron datos.
    """
    # Se omite la descarga si no hay tickers en la lista
    if not tickers_list:
        print("No hay tickers pendientes de actualización. Omitiendo descarga...")
        df_vacio = pd.DataFrame(columns=['Date', 'Ticker', 'Open'])
        return df_vacio, []

    # Asegurar que no existan duplicados
    tickers_unicos = list(set([str(t) for t in tickers_list]))
    
    print(f"Iniciando descarga masiva para {len(tickers_unicos)} tickers únicos...")
    
    # Descarga en bloque (Batch)
    df_batch = yf.download(
        tickers=tickers_unicos, 
        period=periodo, 
        interval=intervalo,
        ignore_tz=False, # se mantiene la zona horaria momentáneamente para limpiarla
        auto_adjust=True # precios ajustados
    )
    
    if df_batch.empty:
        print("No se pudo extraer ningún dato.")
        return pd.DataFrame()

    print("Reestructurando los datos...")
    
    # Transformar de formato "ancho" a "largo"
    # yfinance devuelve un MultiIndex en columnas cuando son varios tickers
    if isinstance(df_batch.columns, pd.MultiIndex):
        # Nombramos los niveles para no perdernos al apilar
        df_batch.columns.names = ['Atributos', 'Ticker']
        
        # .stack() pasa los tickers de las columnas a las filas
        # future_stack=True evita advertencias en versiones nuevas de pandas
        df_prices = df_batch.stack(level='Ticker', future_stack=True).reset_index()
    else:
        # En caso de que la lista tenga solo 1 ticker
        df_prices = df_batch.reset_index()
        df_prices['Ticker'] = tickers_unicos[0]

    # Homogeneizar el nombre de la columna de fecha y quitar la zona horaria
    col_fecha = 'Date' if 'Date' in df_prices.columns else 'Datetime'
    if col_fecha in df_prices.columns:
        # Convertimos a UTC primero para unificar, y luego quitamos la zona horaria
        df_prices[col_fecha] = pd.to_datetime(df_prices[col_fecha], utc=True).dt.tz_localize(None)
        df_prices.rename(columns={'Datetime': 'Date'}, inplace=True)
        df_prices['Date'] = df_prices['Date'].dt.normalize()

    # Eliminar columnas innecesarias
    cols_a_eliminar = ['High', 'Low', 'Capital Gains', 'Stock Splits', 'Adj Close']
    df_prices.drop(columns=cols_a_eliminar, inplace=True, errors='ignore')

    # Eliminar filas donde no hay datos de precio (días no cotizados, IPOs tardíos, etc.)
    df_prices.dropna(subset=['Close'], inplace=True)
    
    # Reiniciar el índice
    df_prices.reset_index(drop=True, inplace=True)   

    # Identificar tickers sin datos
    if not df_prices.empty and 'Ticker' in df_prices.columns:
        tickers_obtenidos = df_prices['Ticker'].unique().tolist()
    else:
        tickers_obtenidos = []
        
    # Restamos los conjuntos: Originales - Obtenidos = Fallidos
    tickers_sin_datos = list(set(tickers_unicos) - set(tickers_obtenidos))
    
    if tickers_sin_datos:
        print(f"Advertencia: No se encontraron datos para {len(tickers_sin_datos)} tickers.") 

    print("Extracción completada.")
    
    # Devolvemos ambos objetos
    return df_prices, tickers_sin_datos


def descargar_constituents_sp(force_update=False):
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


def limpiar_constituents_sp(df:pd.DataFrame)->pd.DataFrame:
    # Seleccionar y renombrar columnas
    df_tickers = df[["Symbol", "Date added"]].copy()
    df_tickers.rename(columns={
        "Symbol": "Ticker",
        "Date added": "DateAdded"
        }, inplace=True)
    
    # Limpieza de tickers
    df_tickers['Ticker'] = df_tickers['Ticker'].map(clean_ticker)

    return df_tickers


def extraer_info(tickers_list:list)->pd.DataFrame:
    """Extrae información sobre los tickers, sin datos historicos."""
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
                'Industry': info.get('industry')
            }
    
            dfs_info.append(row)
    
        except Exception as e:
            print(f"Error con {ticker}: {e}")
            continue
    
    if dfs_info:
        return pd.DataFrame(dfs_info)
    else:
        df_vacio = pd.DataFrame(columns= ['Date', 'Ticker', 'Open'])
        return df_vacio
  

def extraer_financials(tickers_list: list, aproximar_fechas: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Extrae de yfinance datos financieros trimestrales del Estado de Resultados, Balance General y Cash Flow.
    
    Parámetros:
    - tickers_list: Lista de símbolos a extraer.
    - aproximar_fechas: Si es True, usa una estimación estática de días (más rápido). 
                        Si es False, busca las fechas reales de publicación (más lento, evita lookahead bias).
    - Devuelve el dataframe obtenido y la lista de tickers de los cuales no se obtuvieron datos.
    """
    dfs_lista = []
    tickers_sin_datos = []      

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
                tickers_sin_datos.append(ticker)
                continue

            # Se limitan a las primeras 4 columnas (yfinance devuelve 5, la última vacía)
            fin = fin.iloc[:, :4]

            # Transponer y filtrar Estado de Resultados
            df_fin = fin.T.reindex(columns=cols_resultados)

            # Validación del Balance General (si existe)
            if bal is not None and not bal.empty:
                bal = bal.iloc[:, :4] 
                
                # Se renombran las filas para variables con nombres largos (antes de transponer)
                bal = bal.rename(index={
                    'Total Non Current Liabilities Net Minority Interest': 'Total Noncurrent Liabilities',
                    'Total Liabilities Net Minority Interest': 'Total Liabilities'
                })
                
                df_bal = bal.T.reindex(columns=cols_balance)
                
                df_temp = df_fin.join(df_bal, how='left')
            else:
                df_temp = df_fin.copy() 
                # Pandas asigna el float('nan') a toda la lista de columnas de una sola vez
                df_temp[cols_balance] = float('nan')  

            # Validación del Cash Flow (si existe)
            if cf is not None and not cf.empty:
                cf = cf.iloc[:, :4] 
                df_cf = cf.T.reindex(columns=cols_cashflow)
                df_temp = df_temp.join(df_cf, how='left')
            else:
                df_temp[cols_cashflow] = float('nan')       

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

            # --- TRANSFORMACIÓN DE FECHAS #2: ALINEACIÓN MENSUAL ---
            # Se lleva la fecha al dia 1 del mes siguiente
            df_temp['Date'] = df_temp['Date'] + pd.offsets.MonthBegin(1)

            if not df_temp.empty:
                dfs_lista.append(df_temp)
            
            # Pausa aleatoria para no saturar la API
            #time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"Error procesando fundamentales para {ticker}: {e}")
            tickers_sin_datos.append(ticker)
            continue

    # Concatenación final
    if dfs_lista:
        df_final = pd.concat(dfs_lista, axis=0, ignore_index=True)

        # Se agrega columna 'FinancialsSource' para indicar que los datos provienen de yfinance
        df_final['FinancialsSource'] = 'yfinance'

        return df_final, tickers_sin_datos
    else:
        return pd.DataFrame(), tickers_sin_datos


def guardar_tickers_sin_datos(sin_datos_set: set): 
    if not sin_datos_set:
        return

    tickers_totales = set(sin_datos_set)

    if ruta_sin_datos.exists() and ruta_sin_datos.stat().st_size > 0:
        with open(ruta_sin_datos, "r", encoding="utf-8") as f:
            next(f, None)  # Salta la cabecera
            for linea in f:
                ticker = linea.strip()
                if ticker:
                    tickers_totales.add(ticker)

    with open(ruta_sin_datos, "w", encoding="utf-8") as f:
        f.write("Ticker\n")  
        for ticker in sorted(tickers_totales):
            f.write(f"{ticker}\n")


def unir_financials(df_yfinance: pd.DataFrame, df_simfin: pd.DataFrame) -> pd.DataFrame:

    # Retornos tempranos si hay dataframes vacíos
    if df_yfinance.empty and df_simfin.empty:
        print("Ambos DataFrames están vacíos. Omitiendo unión...")
        # Garantizamos la estructura mínima para no romper procesos posteriores
        return pd.DataFrame(columns=['Ticker', 'Date'])
        
    if df_yfinance.empty:
        print("Datos de yfinance vacíos. Se utilizarán solo los datos de SimFin.")
        return df_simfin.drop_duplicates(subset=['Ticker', 'Date'], keep='last').reset_index(drop=True)
        
    if df_simfin.empty:
        print("Datos de SimFin vacíos. Se utilizarán solo los datos de yfinance.")
        return df_yfinance.drop_duplicates(subset=['Ticker', 'Date'], keep='last').reset_index(drop=True)

    # Eliminar posibles duplicados INTERNOS en cada DataFrame antes de cruzarlos
    df_yf_unique = df_yfinance.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
    df_sf_unique = df_simfin.drop_duplicates(subset=['Ticker', 'Date'], keep='last') 

    # Convertir 'Ticker' y 'Date' en el índice temporalmente
    df_yf_idx = df_yf_unique.set_index(['Ticker', 'Date'])
    df_sf_idx = df_sf_unique.set_index(['Ticker', 'Date'])

    # Identificar solapamientos (intersección de índices)
    idx_solapados = df_yf_idx.index.intersection(df_sf_idx.index)

    # Aplicar combine_first para tratar el solapamiento
    # Toma el valor de yfinance, si fuese NaN lo intenta rellenar con SimFin
    df_financials_idx = df_yf_idx.combine_first(df_sf_idx)

    # Ajustar la columna 'FinancialsSource' para registros mezclados
    # Si un registro estaba en ambos, indicamos que contiene datos combinados
    if 'FinancialsSource' in df_financials_idx.columns:

        df_financials_idx.loc[idx_solapados, 'FinancialsSource'] = 'yfinance + simFin'
    
    print(f"Se han encontrado {len(idx_solapados)} filas con Ticker y Date solapados (presentes en ambas fuentes).")

    # Se devuelven 'Ticker' y 'Date' como columnas normales
    df_unido = df_financials_idx.reset_index()

    # Se eliminan filas sin datos financieros: no deberia suceder en esta etapa, se agrega por las dudas
    # Eliminar filas que no tengan ninguna de las columnas críticas 
    columnas_criticas = ['Operating Income', 'EBITDA', 'Net Income', 'Total Assets', 'Total Liabilities']

    # Se asegura buscar solo las columnas que realmente existan en el DataFrame
    # para evitar un KeyError en caso de que alguna no haya sido descargada.
    columnas_presentes = [col for col in columnas_criticas if col in df_unido.columns]
    
    # Se eliminan filas solo si todas las métricas presentes son NaN (how='all')
    if columnas_presentes:
        df_unido = df_unido.dropna(subset=columnas_presentes, how='all')
    

    return df_unido


def limpieza_final(df: pd.DataFrame) -> pd.DataFrame:

    # --- Cláusula de retorno temprano
    if df.empty:
        print("El DataFrame a limpiar está vacío. Omitiendo proceso de limpieza...")
        # Nos aseguramos de mantener la coherencia eliminando espacios en columnas si las hay
        df.columns = df.columns.str.replace(' ', '')
        return df
        
    # Validar que existan las columnas clave antes de ordenar
    if 'Ticker' not in df.columns or 'Date' not in df.columns:
        print("Advertencia: El DataFrame no contiene las columnas 'Ticker' o 'Date'. Abortando limpieza...")
        return df

    # Ordenar cronológicamente
    df = df.sort_values(by=['Ticker', 'Date'])

    # Resetear el índice
    df_clean = df.reset_index(drop=True)

    #  Extraer los nombres de las columnas financieras de yfinance desde el diccionario
    cols_financieras = list(mapa_columnas.values()) + ['FinancialsSource']
    cols_presentes_fin = [col for col in cols_financieras if col in df_clean.columns]
    
    # Se aplica forward fill a las columnas financieras, para completar datos de granularidad mensual
    # Se tolera una demora máxima de 6 meses en la presentación del siguiente reporte trimestral (limit=6)
    if cols_presentes_fin:
        df_clean[cols_presentes_fin] = df_clean.groupby('Ticker')[cols_presentes_fin].ffill(limit=6)

    # Eliminar filas que no tengan datos en las columnas críticas: 
    # precio 'Open' y al menos una métrica financiera clave
    df_clean = df_clean.dropna(subset=['Close'])

    metrica_control = [col for col in ['Total Assets', 'Net Income'] if col in df_clean.columns]
    if metrica_control:
        df_clean = df_clean.dropna(subset=metrica_control, how='all')

    # Eliminar espacios en los nombres de las columnas
    df_clean.columns = df_clean.columns.str.replace(' ', '')

    return df_clean


def guardar_raw_data(df_final_clean:pd.DataFrame)->pd.DataFrame:
    df_datos_nuevos = df_final_clean.copy()

    # Validar si realmente hay datos nuevos para guardar
    if not df_datos_nuevos.empty:
        
        if raw_data_file.exists():
            df_historico = pd.read_parquet(raw_data_file)
            
            # Establecer Ticker y Date como índices para alinear los datos
            df_historico.set_index(['Ticker', 'Date'], inplace=True)
            df_datos_nuevos.set_index(['Ticker', 'Date'], inplace=True)
            
            # combine_first: Prioriza df_datos_nuevos, pero si hay un NaN, usa el valor de df_historico (modelo "Upsert")
            df_consolidado_idx = df_datos_nuevos.combine_first(df_historico)
            
            # Restaurar las columnas normales
            df_consolidado = df_consolidado_idx.reset_index()
            
        else:
            # Si el fichero no existe, el consolidado son los datos actuales
            df_consolidado = df_datos_nuevos

        # Guardar el dataframe completo consolidado
        df_consolidado.to_parquet(raw_data_file, index=False)
        
        print(f"Extracción finalizada.\nDatos guardados exitosamente en '{raw_data_file}'.")
        return df_consolidado
    else:
        print("Extracción finalizada. No se detectaron datos nuevos para agregar.")
        return pd.DataFrame()
    

# --- Función auxiliar para analizar el retardo de publicación ---

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


# --- Bloque principal ---
"""
- Se replica el flujo del Notebook 01_Extraccion.ipynb.
- A veces yfinance devuelve menos datos financieros al ejecutar desde el terminal. 
- Se ejecuta desde la raiz con: python -m src.extract
"""

def main():
    print("\n--- Extracción de datos financieros de simFin ---\n")

    # Extraer datos de simFin
    df_simfin_unfiltered = extraer_simfin()

    # Genera el universo inicial de tickers según los criterios establecidos
    tickers_universe_list = generar_universo_tickers(df_simfin_unfiltered)

    # Filtra los tickers del universo
    df_simfin_filtered = df_simfin_unfiltered[df_simfin_unfiltered['Ticker'].isin(tickers_universe_list)]

    # Estandarizar columnas y fechas para que coincidan con el formato de yfinance
    df_simfin_standarized = estandarizar_simfin(df_simfin_filtered)
    print("Datos extraidos de simFin, dimensiones:", df_simfin_standarized.shape) 


    print("\n--- Extracción de precios de yfinance ---\n")

    # Si existe un fichero raw_data guardado, se verifican los datos que requieran actualización
    new_data_tickers, df_simfin_new_data = gestionar_actualizacion(tickers_universe_list, df_simfin_standarized)

    # Extraer precios y obtener lista de tickers sin precios
    df_prices, tickers_sin_precios = extraer_precios(new_data_tickers)

    # Se extrae precio del Índice de Mercado para usar en cálculos y se guarda en fichero 
    df_index, _ = extraer_precios(['SPY'])
    df_index.to_parquet(f"{data_folder}/market_index.parquet")

    # Se guardan los tickers que obtuvieron precios para filtrar al extraer info y financials de yfinance
    tickers_con_precios_nuevos = df_prices['Ticker'].unique().tolist()

    print(f"Precios extraídos para {len(tickers_con_precios_nuevos)} tickers.")  


    print("\n--- Extracción de datos complementarios ---\n")
  
    # Obtener los componentes del índice S&P 500
    ruta_sp500 = descargar_constituents_sp(force_update=False) 

    # Cargar y limpiar datos
    print("Extrayendo información complementaria")
    df_tickers_sp_raw = pd.read_csv(ruta_sp500)
    df_tickers_sp = limpiar_constituents_sp(df_tickers_sp_raw)

    # Extraer info de Sector e Industry de yfinance
    df_info = extraer_info(tickers_con_precios_nuevos)

    # Se unen los datos complementarios
    df_supplementary = pd.merge(
        df_info,
        df_tickers_sp,
        on= 'Ticker',
        how= 'left'
    )
    print("Unidos los datos complementarios. Dimensiones:", df_supplementary.shape)


    print("\n--- Extracción de datos financieros de yfinance ---\n")

    print("Extrayendo datos financieros de yfinance, puede demorar varios minutos.")
    df_yfinance, tickers_sin_financials = extraer_financials(
        tickers_con_precios_nuevos, 
        aproximar_fechas = False
        )
    print("Extracción de financials de yfinance finalizada. Dimensiones:", df_yfinance.shape)


    print("\n--- Unión de datasets y almacenamiento ---\n")

    # Se guardan los tickers sin datos en fichero .csv para descartarlos si se itera la extracción
    sin_datos_set = set(tickers_sin_precios+tickers_sin_financials)
    guardar_tickers_sin_datos(sin_datos_set)

    df_financials_completo = unir_financials(df_yfinance, df_simfin_new_data)  
    print("Unidos datasets de financials. Dimensiones:", df_financials_completo.shape)

    # Unir precios con datos financieros
    df_merged = pd.merge(
        df_prices, 
        df_financials_completo, 
        on=['Date', 'Ticker'],
        how='left' 
    )
    print("Unidos datasets de precios y financials. Dimensiones:", df_merged.shape)

    # Se agregan las columnas de info complementaria
    df_final = pd.merge(
        df_merged, 
        df_supplementary, 
        on=['Ticker'],
        how='left' 
    )
    print("Columnas de info complementaria agregadas. Dimensiones:", df_final.shape)

    # Limpieza final
    df_final_clean = limpieza_final(df_final)
    print("Dimensiones del dataset final limpio:", df_final_clean.shape)

    # Guardar/actualizar fichero raw_data
    df_guardado = guardar_raw_data(df_final_clean)
    df_guardado.info()
    print("Tickers sin datos:", len(sin_datos_set))
    print("Tickers guardados:", len(df_guardado['Ticker'].unique().tolist()))    

if __name__ == "__main__":
    main()


# Función "legacy": extraer datos macro de Fred
"""
Ya no se utiliza en el código actual por su baja relevancia en la predicción,
debido a que los valores son los mismos para todos los tickers.

La dejo aquí por las dudas.
"""

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

'''