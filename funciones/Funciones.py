# Funciones
import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Funciones ETL

def extraer_precios(tickers_list:list)->pd.DataFrame:
    dfs_prices = []
    for ticker in tickers_list:
        df = yf.Ticker(ticker).history(period='1d')
        df['Ticker'] = ticker
        dfs_prices.append(df)  
        
    # Concatenar en un dataframe
    df_prices = pd.concat(dfs_prices, ignore_index = True)
    
    # Quitar columnas
    df_prices.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1, inplace= True)

    return df_prices

def extraer_info(tickers_list:list)->pd.DataFrame:
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
    
    df = pd.DataFrame(dfs_info)
    return df

def extraer_financials(tickers_list:list)->pd.DataFrame:
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
            df_temp = df_temp.reindex(columns=columnas)
            df_temp['Ticker'] = ticker

            dfs_financials.append(df_temp)

        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            continue

    if dfs_financials:
        return pd.concat(dfs_financials, axis=0)
    else:
        return pd.DataFrame()


# Gestion de outliers
    
def gestiona_outliers(col,clas = 'check'):
    
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
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

# Graficos

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
    
## Función para gráfico de barras de variables categóricas
def cat_plot(col):
     if col.dtypes == 'category':
        fig = px.bar(col.value_counts())
        #fig = sns.countplot(x=col)
        return(fig)


## Función general plot para aplicar al archivo por columnas
def plot(col):
     if col.dtypes != 'category':
        print('Cont')
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_plot(col).show()

# Modelizar

def cramers_v(var1, varObj):

    # --- var1 ---
    if pd.api.types.is_numeric_dtype(var1):
        var1 = pd.cut(var1, bins=5)
    else:
        var1 = var1.astype('category')

    # --- varObj ---
    if pd.api.types.is_numeric_dtype(varObj):
        varObj = pd.cut(varObj, bins=5)
    else:
        varObj = varObj.astype('category')

    tabla = pd.crosstab(var1, varObj)
    vCramer = stats.contingency.association(tabla.values, method='cramer')
    return vCramer


# Función para generar la fórmula por larga que sea
def ols_formula(df, dependent_var, *excluded_cols):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)