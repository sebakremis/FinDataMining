# funciones.py
import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import periodo, intervalo

# Funciones ETL

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


def extraer_datos_fundamentales(tickers_list: list) -> pd.DataFrame:
    """
    Extrae el Estado de Resultados y el Balance General de una lista de tickers,
    y devuelve un DataFrame unificado ideal para cálculo de ratios históricos.
    """
    dfs_lista = []
    
    # Separamos las columnas según de qué reporte provienen
    cols_resultados = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 
                   'EBITDA', 'Basic Average Shares'] 

    cols_balance = ['Cash And Cash Equivalents', 'Current Debt', 'Long Term Debt', 
                'Total Debt', 'Stockholders Equity', 'Total Assets', 'Current Assets', 'Current Liabilities'] 

    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # 1. Extraer ambos reportes anuales
            fin = yf_ticker.financials
            bal = yf_ticker.balance_sheet

            # Validación del Estado de Resultados
            if fin is None or fin.empty:
                print(f"Sin datos financieros para {ticker}")
                continue

            # 2. Transponer y filtrar Estado de Resultados
            df_fin = fin.T.reindex(columns=cols_resultados)

            # 3. Transponer y filtrar Balance General (si existe)
            if bal is not None and not bal.empty:
                df_bal = bal.T.reindex(columns=cols_balance)
                # Unimos ambos reportes usando la fecha (que actualmente es el índice)
                df_temp = df_fin.join(df_bal, how='left')
            else:
                # Si no hay balance, nos quedamos con financials y llenamos el resto con nulos
                df_temp = df_fin
                for col in cols_balance:
                    df_temp[col] = pd.NA

            # 4. Limpieza del DataFrame temporal
            df_temp = df_temp.reset_index() # Pasamos la fecha del índice a una columna
            df_temp = df_temp.rename(columns={'index': 'Fecha'})
            df_temp['Ticker'] = ticker # Identificador del activo
            
            dfs_lista.append(df_temp)

        except Exception as e:
            print(f"Error procesando fundamentales para {ticker}: {e}")
            continue

    # 5. Concatenación final
    if dfs_lista:
        # ignore_index=True es clave aquí para tener un índice numérico limpio (0, 1, 2...)
        df_final = pd.concat(dfs_lista, axis=0, ignore_index=True)
        # Opcional: Asegurar que la fecha tenga formato datetime
        df_final['Fecha'] = pd.to_datetime(df_final['Fecha']).dt.tz_localize(None)
        return df_final
    else:
        return pd.DataFrame()


def extraer_financials(tickers_list:list)->pd.DataFrame:
    """
    Extrae información financiera de una lista de tickers y devuelve un DataFrame con los datos seleccionados.
    """
    dfs_financials = []
    columnas = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
    for ticker in tickers_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            financials_data = yf_ticker.financials

            # Validación
            if financials_data is None or financials_data.empty:
                print(f"Sin datos para {ticker}")
                continue

            df_temp = financials_data.T 
            # df_temp = df_temp.iloc[[0]]  # Solo la última fila disponible
            df_temp = df_temp.reindex(columns=columnas)
            df_temp['Ticker'] = ticker

            dfs_financials.append(df_temp)

        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            continue

    if dfs_financials:
        return pd.concat(dfs_financials, axis=0, ignore_index=False)
    else:
        return pd.DataFrame()


def extraer_info(tickers_list:list)->pd.DataFrame:
    """Extrae información fundamental, sin datos historicos."""
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

# Cálculos de métricas financieras y ratios de valuación

def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame limpio de precios y fundamentales alineados, 
    y calcula métricas de valoración históricas.
    """
    # Trabajamos sobre una copia para no alterar el original inadvertidamente
    df_metrics = df.copy()

    # --- 1. DATOS BASE ---
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

    # --- 2. RATIOS DE VALORACIÓN (Mercado vs Contabilidad) ---
    df_metrics['PE_Trailing'] = df_metrics['MarketCap'] / df_metrics['Net Income']
    df_metrics['EnterpriseToEbitda'] = df_metrics['EnterpriseValue'] / df_metrics['EBITDA']
    
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['PriceToBook'] = df_metrics['MarketCap'] / df_metrics['Stockholders Equity']

    # --- 3. RATIOS DE RENTABILIDAD Y MÁRGENES ---
    df_metrics['operatingMargins'] = df_metrics['Operating Income'] / df_metrics['Total Revenue']
    df_metrics['profitMargins'] = df_metrics['Net Income'] / df_metrics['Total Revenue']
    
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['returnOnEquity'] = df_metrics['Net Income'] / df_metrics['Stockholders Equity']
        
    if 'Total Assets' in df_metrics.columns:
        df_metrics['ReturnOnAssets'] = df_metrics['Net Income'] / df_metrics['Total Assets']

    # --- 4. RATIOS DE LIQUIDEZ Y SOLVENCIA ---
    if 'Stockholders Equity' in df_metrics.columns:
        df_metrics['debtToEquity'] = deuda_total / df_metrics['Stockholders Equity']
        
    if 'Current Assets' in df_metrics.columns and 'Current Liabilities' in df_metrics.columns:
        df_metrics['currentRatio'] = df_metrics['Current Assets'] / df_metrics['Current Liabilities']

    # --- LIMPIEZA FINAL ---
    # Redondear para legibilidad y consistencia
    cols_a_redondear = [
        'PE_Trailing', 'PriceToBook', 'EnterpriseToEbitda', 'operatingMargins', 
        'profitMargins', 'returnOnEquity', 'ReturnOnAssets', 'debtToEquity', 'currentRatio'
    ]
    
    # Aplicar redondeo solo a las columnas que realmente existen en el df_metrics
    cols_presentes = [col for col in cols_a_redondear if col in df_metrics.columns]
    df_metrics[cols_presentes] = df_metrics[cols_presentes].round(4) # 4 decimales para capturar bien los porcentajes

    return df_metrics


def calcular_betas(df_precios: pd.DataFrame, df_index: pd.DataFrame, ventana_semanas: int = 52) -> pd.DataFrame:
    """
    Calcula el Beta dinámico (móvil) para cada fecha basándose en una ventana 
    de tiempo previa (por defecto 52 semanas).
    """
    ticker_mercado = df_index['Ticker'].iloc[0]
    
    # 1. Unir y pivotear para tener fechas alineadas
    df_unido = pd.concat([df_precios, df_index], ignore_index=True)
    df_pivot = df_unido.pivot(index='Fecha', columns='Ticker', values='Close')
    
    # 2. Calcular retornos porcentuales semanales
    df_retornos = df_pivot.pct_change()
    
    # 3. Separar los retornos del mercado y calcular su Varianza Móvil
    retornos_mercado = df_retornos[ticker_mercado]
    varianza_mercado_movil = retornos_mercado.rolling(window=ventana_semanas).var()
    
    dfs_betas_historicos = []
    
    # 4. Iterar sobre cada acción
    for ticker in df_retornos.columns:
        if ticker == ticker_mercado:
            continue
            
        retornos_activo = df_retornos[ticker]
        
        # Calcular la Covarianza Móvil entre la acción y el mercado
        covarianza_movil = retornos_activo.rolling(window=ventana_semanas).cov(retornos_mercado)
        
        # Calcular el Beta Móvil (Fórmula: Covarianza / Varianza)
        beta_movil = covarianza_movil / varianza_mercado_movil
        
        # Estructurar los resultados de vuelta a formato de columnas
        df_temp = pd.DataFrame({
            'Fecha': df_retornos.index,
            'Ticker': ticker,
            'Beta': beta_movil
        })
        dfs_betas_historicos.append(df_temp)
        
    # 5. Concatenar todos los tickers
    df_betas_final = pd.concat(dfs_betas_historicos, ignore_index=True)
    
    # Redondear para que quede limpio
    df_betas_final['Beta'] = df_betas_final['Beta'].round(4)
    
    return df_betas_final

# Gestion de outliers
    
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

# Modelizar

def obtener_metricas(y_real, y_pred, nombre_modelo):
    """
    Calcula métricas de evaluación para modelos de regresión.
    """
    mse = mean_squared_error(y_real, y_pred)
    
    
    return {
        'Modelo': nombre_modelo,
        'MAE': mean_absolute_error(y_real, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_real, y_pred)
    }

# Bloque principal para pruebas desde el terminal

def main():
    pass


if __name__ == "__main__":
    main()