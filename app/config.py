import chromadb
import cohere
import os
import math
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from challengefinal_schemas import ProductInput, CustomerQuestion

# =========================
# CONFIGURACIÓN
# =========================

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

EMBED_MODEL = "embed-multilingual-v3.0"
SIMILARITY_THRESHOLD = 0.5

app = FastAPI(
    title="Challenge Final Ecommerce WhatsApp API",
    version="1.0.0"
)

# =========================
# CHROMA DB
# =========================

chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="./chroma_products"
    )
)

collection = chroma_client.get_or_create_collection(
    name="products"
)

# =========================
# BASES EN MEMORIA
# =========================

PRODUCT_DB = {}
CONVERSATIONS_DB = {}
MAX_HISTORY = 10
