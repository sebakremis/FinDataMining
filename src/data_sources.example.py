"""
src/data_sources.example.py
Ejemplo de archivo para la gestión de claves API DE SimFin.

-- NO ESCRIBAS tu clave real en este archivo. 

-- Asegúrate de mantener tanto .env como /src/data_sources.py 
en tu .gitignore para evitar subir tus claves a repositorios públicos.


Después de clonar:
1. Crea un archivo llamado data_sources.py en la carpeta src, copiando este archivo.
2. Crea un archivo .env en la raíz del proyecto con la siguiente linea, reemplazando tu_clave_real_aqui por tu clave real:

SIMFIN_API_KEY="tu_clave_real_aqui"


"""

"""
src/data_sources.py
Modulo privado para gestionar la carga de claves API.
NO ESCRIBAS tu clave real en este archivo. 
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Se obtiene la clave API desde las variables de entorno
simfin_api_key = os.getenv("SIMFIN_API_KEY")