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
from src.data_sources import simfin_api_key
from src.config import (
    periodo, intervalo, cols_resultados, cols_balance,
    cols_cashflow, data_folder, mapa_columnas,
    retardo_publicacion, tickers_file, cambios_tickers
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


def inicializar_datos_simfin(update_tickers:bool=True)->pd.DataFrame:
    temp_file = Path(data_folder) / "temp_data.parquet"

    if not tickers_file.is_file() or update_tickers:
        if not update_tickers:
            print("No se encuentra el fichero de tickers.\nIndicar update_tickers=True para volver a generarlo.\n")
            return pd.DataFrame()

        print("Generando universo de tickers...\nSe extraen datos de simFin:")
        df = extraer_simfin()
        generar_universo_tickers(df)
        # Guardar datos en fichero temporal
        df.to_parquet(temp_file)
        
    if temp_file.is_file():
        df = pd.read_parquet(temp_file)
    else:
        print("Se extraen datos de simFin:")
        df = extraer_simfin()
    
    # Eliminar fichero temporal
    if temp_file.is_file():
        temp_file.unlink()

    return df


def generar_universo_tickers(df: pd.DataFrame, 
                             umbral_filas: int = 19, 
                             columna_ranking: str = 'Revenue', 
                             cantidad_tickers: int = 550):
    """
    Se genera el fichero del universo de tickers.
    1. Filtra los tickers que superen el umbral de filas.
    2. Agrupa por ticker y calcula el promedio de su métrica para ranquearlos.
    3. Selecciona el top N y guarda en CSV.
    """
    # Limpiar tickers
    df['Ticker'] = df['Ticker'].map(clean_ticker)

    # Contar y filtrar tickers válidos
    counts = df.groupby("Ticker").size()
    tickers_validos = counts[counts >= umbral_filas].index
    df_filtrado = df[df["Ticker"].isin(tickers_validos)]
    
    # Calcular un valor representativo de 'Revenue' por empresa.
    ranking_metricas = df_filtrado.groupby("Ticker")[columna_ranking].mean()
    
    # Se ordenan de mayor a menor y tomar el Top
    top_tickers = ranking_metricas.sort_values(ascending=False).head(cantidad_tickers).index.tolist()

    # Se modifican los tickers que hayan cambiado de nombre
    tickers_corregidos = [cambios_tickers.get(t, t) for t in top_tickers]

    # Quitar si existen duplicados
    tickers_unicos = list(set([str(t) for t in tickers_corregidos]))
    
    # Guardar fichero CSV
    df_salida = pd.DataFrame({"Ticker": tickers_unicos})
    df_salida.to_csv(tickers_file, index=False)
    
    print(f"Fichero de tickers guardado exitosamente. Universo total: {len(tickers_unicos)} tickers.")


def extraer_precios(tickers_list: list) -> pd.DataFrame:
    """
    Extrae precios históricos en lote y formatea las fechas para cruzar con datos fundamentales.
    """
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

    print("Extracción completada.")
    return df_prices


def actualizar_universo_tickers(df:pd.DataFrame)->list:
    # Obtener los tickers únicos y convertirlos a una lista
    tickers_universe_list = df['Ticker'].unique().tolist()

    # Recrear el dataframe de tickers
    df_tickers_universe = pd.DataFrame({'Ticker': tickers_universe_list})

    # Guardar los tickers actualizados
    df_tickers_universe.to_csv(tickers_file, index=False)

    return tickers_universe_list


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
        return pd.DataFrame()


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
            tickers_sin_datos.append(ticker)
            continue

    # Concatenación final
    if dfs_lista:
        df_final = pd.concat(dfs_lista, axis=0, ignore_index=True)
        return df_final, tickers_sin_datos
    else:
        return pd.DataFrame(), tickers_sin_datos


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

    # Se seleccionan los tickers del S&P500
    #df_final = df_consolidado[df_consolidado['Ticker'].isin(tickers_validos)]
    
    return df_consolidado


def estandarizar_simfin(df_raw:pd.DataFrame, cols:list) -> pd.DataFrame:
    df = df_raw.copy()
    df.rename(columns=mapa_columnas, inplace=True)

    # Se crean las columnas faltantes vacías, luego serán calculadas en la fase de transformación
    cols_faltantes = ['EBITDA', 'Total Debt', 'Free Cash Flow']
    df[cols_faltantes] = np.nan

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
    hoy = pd.Timestamp.today().normalize()
    df = df[df['Date'] <= hoy]

    # Ordenar cronológicamente
    df = df.sort_values(by=['Ticker', 'Date'])

    # Eliminar las filas anteriores al primer reporte financiero disponible
    columna_critica = 'Operating Income' 
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


# Bloque principal (Se replica el flujo del Notebook Extraccion)
# Se ejecuta desde la raiz: python -m src.extract

def main():
    # Importar librerías
    import pandas as pd
    from src.config import data_folder, tickers_file

    # Extraer datos de simFin
    df_simfin_unfiltered = inicializar_datos_simfin(update_tickers=True)
    print("Datos extraidos de simFin, dimensiones:", df_simfin_unfiltered.shape)

    # Se lee el fichero con los tickers incluidos en el proyecto
    df_tickers_universe = pd.read_csv(tickers_file)
    tickers_universe_list = df_tickers_universe['Ticker'].tolist()
    tickers_unique = list(set(tickers_universe_list)) # se asegura que no hayan duplicados
    print("Universo de tickers extraido, cantidad de tickers:",len(tickers_unique))

    # Extraer precios de los tickers y del índice SPY 
    print("Extrayendo precios de yfinance, demora unos minutos.")
    df_prices = extraer_precios(tickers_unique)

    # Se extrae precio del Índice de Mercado para usar en cálculos y se guarda en fichero 
    df_index = extraer_precios(['SPY'])
    df_index.to_parquet(f"{data_folder}/market_index.parquet")

    print("Extraidos precios del universo de tickers y del indice.")
    print("Dimensiones de df_precios:", df_prices.shape)
    print("Dimensiones de df_index:", df_index.shape)

    # Actualizar el universo de tickers con los que se obtuvieron precios
    tickers_list_updated = actualizar_universo_tickers(df_prices)
    print("Actualizado el universo a tickers con precios disponibles en yfinance.")
    print("Cantidad de tickers actualizado:", len(tickers_list_updated))

    # Filtrar datos de simFin
    df_simfin = df_simfin_unfiltered[df_simfin_unfiltered['Ticker'].isin(tickers_list_updated)]

    # Obtener constituents del indice S&P 500
    ruta_sp500 = descargar_constituents_sp(force_update=False) 

    # Cargar y limpiar datos de constituents
    df_tickers_sp_raw = pd.read_csv(ruta_sp500)
    df_tickers_sp = limpiar_constituents_sp(df_tickers_sp_raw)
    print("Cargados los tickers del S&P 500 del fichero.")
    print("Cantidad de tickers:", df_tickers_sp['Ticker'].nunique())

    # Unir df_prices y df_tickers
    df_merged = pd.merge(
        df_prices,
        df_tickers_sp,
        on= 'Ticker',
        how= 'left'
    )
    print("Unidos precios con DateAdded del indice S&P500, dimensiones del dataset:", df_merged.shape)

    # Extraer info de sector e industria
    print("Extrayendo info de yfinance. Demora unos minutos.")
    df_info = extraer_info(tickers_list_updated)
    print("Extracción de info finalizada, dimensiones:",df_info.shape)

    # Se unen los dataframes
    df_with_info = pd.merge(
        df_merged,
        df_info,
        on= 'Ticker',
        how= 'left'
    )
    print("Unidos precios e info, dimensiones del dataset:", df_with_info.shape)

    # Extraer datos financieros de yfinance: ultimos 4 trimestres
    print("Extrayendo datos financieros de yfinance, demora varios minutos.")
    df_yfinance, tickers_sin_datos = extraer_financials(tickers_list_updated, aproximar_fechas = True)
    print("Extracción de financials de yfinance finalizada, dimensiones del dataset:", df_yfinance.shape)


    # Definir columnas a mantener en simFin para que coincidan y estandarizar antes de unir
    cols_yfinance = df_yfinance.columns
    df_simfin_clean = estandarizar_simfin(df_simfin, cols_yfinance)

    df_financials_completo = unir_financials(df_yfinance, df_simfin_clean)
    print("Unidos datasets de financials de simFin y yfinance, dimensiones:", df_financials_completo.shape)

    # Unir dataframe de precios con datos financieros
    df_final = pd.merge(
        df_with_info, 
        df_financials_completo, 
        on=['Date', 'Ticker'],
        how='left'
    )
    print("Unidos datasets de precios y financials, dimensiones:", df_final.shape)

    # Limpieza final
    df_final_clean = limpieza_final(df_final)

    # Se quitan del dataset los tickers sin datos financieros de yfinance
    tickers_unicos = set(tickers_sin_datos) # se convierte en set primero por si hay tickers duplicados
    df_final_clean = df_final_clean[~df_final_clean['Ticker'].isin(tickers_unicos)].reset_index(drop=True)
    print(f"Eliminados tickers sin datos financieros en yfinance, quedan {len(tickers_unicos)} tickers.")
    print("Dimensiones del dataset actualizado:", df_final_clean.shape)

    # Guardar datos extraidos en fichero raw_data
    df_final_clean.to_parquet(f"{data_folder}/raw_data.parquet")
    print("Extracción finalizada.\nFichero 'raw_data.parquet' guardado en la carpeta",data_folder)

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