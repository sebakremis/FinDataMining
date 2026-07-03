# src/config.py
from pathlib import Path

# Establecer ubicación de datos
data_folder = "data"
raw_data_file = Path(data_folder) / "raw_data.parquet"
ruta_sin_datos = Path(data_folder) / "tickers_sin_datos.csv"

# Definir periodo de extracción e intervalo de precios
periodo = '6y'  
intervalo = '1mo'

# Mapeo de tickers que cambiaron de nombre
cambios_tickers = {
    'ATUS': 'OPTU',  # Altice USA cambió a Optimum Communications (Nov 2025)
    'AXL': 'DCH',    # American Axle cambió a Dauch Corporation (Ene 2026)
    'RLGY': 'HOUS',  # Realogy Holdings cambió a Anywhere Real Estate
    'RCII': 'UPBD',  # Rent-A-Center cambió a Upbound Group
    'GPS': 'GAP',    # Gap Inc. cambió su ticker a GAP
    'IIVI': 'COHR',  # II-VI Inc. cambió su nombre a Coherent
}

# Definir la regla del retardo de publicación en días
# Se usa para estimar en datos financieros de yfinance que no la incluyen
retardo_publicacion = 30

# Mapeo de columnas esperadas en los reportes financieros (key = simfin, value = yfinance)
# Faltan 3 columnas que se calculan en funcionesExtract.estandarizar_simfin()
mapa_columnas = {
    # --- RESULTADOS (Income Statement) ---
    'Revenue': 'Total Revenue',
    'Gross Profit': 'Gross Profit',
    'Operating Income (Loss)': 'Operating Income',
    'Net Income': 'Net Income',
    'Shares (Basic)': 'Basic Average Shares',

    # --- BALANCE (Balance Sheet) ---
    'Cash, Cash Equivalents & Short Term Investments': 'Cash And Cash Equivalents',
    'Short Term Debt': 'Current Debt',
    'Long Term Debt': 'Long Term Debt',
    'Total Equity': 'Stockholders Equity',
    'Total Assets': 'Total Assets',
    'Total Current Assets': 'Current Assets',
    'Total Current Liabilities': 'Current Liabilities',
    'Total Noncurrent Liabilities': 'Total Noncurrent Liabilities',
    'Total Liabilities': 'Total Liabilities',

    # --- FLUJO DE CAJA (Cash Flow) ---
    'Net Cash from Operating Activities': 'Operating Cash Flow',
    'Net Cash from Investing Activities': 'Investing Cash Flow',
    'Net Cash from Financing Activities': 'Financing Cash Flow',
    'Change in Fixed Assets & Intangibles': 'Capital Expenditure',
    'Depreciation & Amortization': 'Depreciation And Amortization'
}

# Listas de columnas por sector
# Necesario para distinguir tipo de columna al calcular valores TTM en Transform
cols_resultados = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 
                    'EBITDA', 'Basic Average Shares'] 

cols_balance = ['Cash And Cash Equivalents', 'Current Debt', 'Long Term Debt', 
                'Total Debt', 'Stockholders Equity', 'Total Assets', 'Current Assets', 
                'Current Liabilities','Total Noncurrent Liabilities',
                'Total Liabilities']

cols_cashflow = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 
                    'Free Cash Flow', 'Capital Expenditure', 'Depreciation And Amortization'] 