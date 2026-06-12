"""
src/data_sources.example.py
Ejemplo de archivo para la gestión de claves API de FRED.

-- No escribas tu clave real en este archivo. 

Después de clonar:
1. Crea un archivo llamado data_sources.py en la carpeta src, copiando este archivo.
2. Crea un archivo .env en la raíz del proyecto con la siguiente linea, reemplazando tu_clave_real_aqui por tu clave de FRED:

FRED_API_KEY="tu_clave_real_aqui"

-- Asegúrate de mantener tanto .env como /src/data_sources.py en tu .gitignore para evitar subir tus claves a repositorios públicos.
"""

"""
src/data_sources.py
Modulo privado para gestionar la carga de claves API.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Se obtiene la clave API para acceder a datos macro de la Reserva Federal (FRED) desde las variables de entorno
fred_api_key = os.getenv("FRED_API_KEY")