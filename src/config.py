# src/config.py

# Establecer carpeta de datos para todo el repo. 
# Modificar para almacenar los datos en otra ubicación.
data_folder = "data"

# Definir periodo de extracción e intervalo de precios
periodo = '6y'  
intervalo = '1mo'  

# Mapeo de columnas esperadas en los reportes financieros (key = simfin, value = yfinance)

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

    # --- FLUJO DE CAJA (Cash Flow) ---
    'Net Cash from Operating Activities': 'Operating Cash Flow',
    'Net Cash from Investing Activities': 'Investing Cash Flow',
    'Net Cash from Financing Activities': 'Financing Cash Flow',
    'Change in Fixed Assets & Intangibles': 'Capital Expenditure'
}


cols_resultados = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 
                    'EBITDA', 'Basic Average Shares'] 

cols_balance = ['Cash And Cash Equivalents', 'Current Debt', 'Long Term Debt', 
                'Total Debt', 'Stockholders Equity', 'Total Assets', 'Current Assets', 
                'Current Liabilities']

cols_cashflow = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 
                    'Free Cash Flow', 'Capital Expenditure'] 