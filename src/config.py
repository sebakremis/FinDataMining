# src/config.py

# Definir periodo de extracción de datos e intervalo de precios
periodo = '4y'  # Últimos 4 años
intervalo = '1mo'  # Precios mensuales

# Columnas esperadas para cada reporte financiero:
cols_resultados = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 
                    'EBITDA', 'Basic Average Shares'] 

cols_balance = ['Cash And Cash Equivalents', 'Current Debt', 'Long Term Debt', 
                'Total Debt', 'Stockholders Equity', 'Total Assets', 'Current Assets', 
                'Current Liabilities']

cols_cashflow = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 
                    'Free Cash Flow', 'Capital Expenditure'] 