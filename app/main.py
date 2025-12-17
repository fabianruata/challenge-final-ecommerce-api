import math
import chromadb
from fastapi import FastAPI, HTTPException
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.schemas import ProductInput, CustomerQuestion
from app.config import (
    co,
    EMBED_MODEL,
    SIMILARITY_THRESHOLD,
    MAX_HISTORY
)

# =========================
# APP
# =========================

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
# MEMORIA
# =========================

PRODUCT_DB = {}
CONVERSATIONS_DB = {}

# =========================
# UTILIDADES
# =========================

def embed_texts(texts: List[str]):
    response = co.embed(
        model=EMBED_MODEL,
        texts=texts,
        input_type="search_document"
    )
    return response.embeddings


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return round(dot / (norm1 * norm2), 2)


def clean_text(text: str) -> str:
    return (
        text.replace("\\n", " ")
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
    )


def add_message(phone: str, role: str, content: str):
    CONVERSATIONS_DB.setdefault(phone, []).append({
        "role": role,
        "content": content
    })


def get_last_messages(phone: str):
    return CONVERSATIONS_DB.get(phone, [])[-MAX_HISTORY:]

# =========================
# ENDPOINTS
# =========================

@app.post("/products")
def add_products(products: List[ProductInput]):

    if not products:
        raise HTTPException(400, "La lista de productos está vacía")

    chunks, ids, metadatas = [], [], []

    for product in products:

        if product.codigo in PRODUCT_DB:
            raise HTTPException(
                400,
                f"El producto con código {product.codigo} ya existe"
            )

        text = (
            f"Codigo: {product.codigo}\n"
            f"Descripcion: {product.descripcion}\n"
            f"Caracteristicas: {product.caracteristicas}\n"
            f"Precio: {product.precio_venta}"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append(chunk)
            ids.append(f"{product.codigo}_{i}")
            metadatas.append({"codigo": product.codigo})

        PRODUCT_DB[product.codigo] = product

    embeddings = embed_texts(chunks)

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {"message": "Productos cargados correctamente"}

# =========================
# CONSULTA CLIENTE
# =========================

@app.post("/ask")
def ask_product(data: CustomerQuestion):

    # 1. Embedding de la pregunta
    query_embedding = embed_texts([data.pregunta])[0]

    stored = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    context_chunks = []

    for i in range(len(stored["documents"])):
        score = cosine_similarity(
            query_embedding,
            stored["embeddings"][i]
        )

        if score >= SIMILARITY_THRESHOLD:
            context_chunks.append(
                clean_text(stored["documents"][i])
            )

    # 2. Fallback
    if not context_chunks:
        respuesta = clean_text(
            f"Hola {data.nombre_apellido}, somos tienda virtual, trabajamos por encargo directo de fábricas a su domicilio"
            "En este momento no tenemos el producto que está buscando"
            "Cuenteme un poco más así veo si puedo ayudarle"
        )

        add_message(data.telefono, "user", data.pregunta)
        add_message(data.telefono, "assistant", respuesta)

        return {"respuesta": respuesta}

    # 3. Contexto limpio
    context = " ".join(context_chunks)

    # 4. Historial limpio
    history = get_last_messages(data.telefono)

    history_text = ""
    for msg in history:
        role = "Cliente" if msg["role"] == "user" else "Vendedor"
        history_text += f"{role}: {clean_text(msg['content'])} "

    context = "\n".join(
        f"- {chunk}" for chunk in context_chunks
    )


    # 5. Prompt final
    prompt = f"""
{SYSTEM_PROMPT}

Historial reciente de la conversación:
{history_text}

Productos disponibles relevantes:
{context}

Mensaje actual del cliente:
{clean_text(data.pregunta)}

Respuesta del vendedor (natural, clara y estilo WhatsApp):
"""

    # 6. LLM
    response = co.chat(
        model="command-a-03-2025",
        message=prompt
    )

    answer = clean_text(response.text)

    # 7. Guardar historial
    add_message(data.telefono, "user", data.pregunta)
    add_message(data.telefono, "assistant", answer)

    return {
        "respuesta": answer
    }
