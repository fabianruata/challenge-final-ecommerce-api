import os
from dotenv import load_dotenv
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("Falta definir COHERE_API_KEY en el archivo .env")

co = cohere.Client(COHERE_API_KEY)

EMBED_MODEL = "embed-multilingual-v3.0"
SIMILARITY_THRESHOLD = 0.5
MAX_HISTORY = 10
