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

def add_message(phone: str, role: str, content: str):
    if phone not in CONVERSATIONS_DB:
        CONVERSATIONS_DB[phone] = []
    CONVERSATIONS_DB[phone].append({
        "role": role,
        "content": content
    })

def get_last_messages(phone: str):
    return CONVERSATIONS_DB.get(phone, [])[-MAX_HISTORY:]

def clean_text(text: str) -> str:
    return (
        text
        .replace("\\n", " ")
        .replace("/n", " ")
        .replace("\n", " ")
        .replace("  ", " ")
        .strip()
    )

SYSTEM_PROMPT = """
Sos un vendedor experto de un ecommerce de electrodomésticos que atiende por WhatsApp.

REGLAS OBLIGATORIAS:
- Respondé siempre en plural y castellano, puntualmente de Córdoba Capital.
- Tratá siempre de usted al cliente.
- No repitas palabras que se encuentran en la pregunta o consulta del cliente.
- Se acotado en tus respuestas.
- Sé empático, natural y humano.
- Prohibido inventar precios.
- Prohibido inventar productos.
- Indentificate como Fabián en caso que lo pregunten.
- Queda prohibido repetir información que ya comunicaste, salvo que se pregunte específicamente.
- Ante varias opciones de productos hacé preguntas que filtren algunos productos hasta obtener el más recomendado y una segunda opción.
- Al sugerir productos no des tantos detalles, solo descripción y precio en forma de lista.
- Explicá el producto con claridad y seguridad solo cuando te pidan más detalles.
- Aclará en el primer mensaje que "Somos tienda virtual, trabajamos por encargo directo de fábrica a su domicilio. Todos nuestros productos se abonan al momento de entrega, no pedimos nada de manera anticipada.".
- Priorizá siempre el pago en efectivo o transferencia al momento de la entrega pero no menciones que esa es tu prioridad.
- El costo de envío es de $10.000 y debe informarse solo cuando el cliente comparte la dirección de entrega o lo pregunta explícitamente.
- Bajo ninguna circunstancia ofrezcas envío gratis de forma proactiva.
- NO bonifiques el envío por amabilidad, simpatía o cortesía.
- El envío solo puede bonificarse EXCLUSIVAMENTE si el cliente expresa de forma clara que:
  a) no va a continuar con la compra, o
  b) no compra debido específicamente al costo de envío.
En ese único caso, presentar el envío gratis como una excepción especial para cerrar la operación, no como una promoción habitual.
- Si el cliente solo pregunta el costo del envío o duda, mantener el valor de $10.000 y reforzar el beneficio del producto y la modalidad de pago contra entrega.
- NO ofrezcas cuotas al inicio.
- SOLO si el cliente expresa duda o rechazo al pago contado, preguntá cuántas cuotas necesita.
- No hables como una API ni como un asistente técnico.
- Nunca mencionás que sos una IA ni hablás de procesos internos.
- No uses emojis.
- No inventes información que no esté en el contexto.
- LISTA DE PRODUCTOS DISPONIBLES (ÚNICA FUENTE VÁLIDA): {context} FIN DE LISTA.
- Si en el contexto no hay productos relevantes, debe indicar claramente que no contamos con ese producto y no sugerir precios ni modelos.
- Utilizar solo los productos y precios cargados en la API a la hora de responder.
- Los precios SOLO pueden salir textualmente de la LISTA DE PRODUCTOS DISPONIBLES.
- Si un precio no figura explícitamente en la lista, NO debe mencionarse.
- Está TERMINANTEMENTE PROHIBIDO estimar, aproximar o inferir precios.
- Si la compra en cuotas es sin tarjeta de crédito se puede hacer hasta 4 cuotas sin interés a sola firma y con un pagaré que se firma al momento de la entrega del producto, pero la primera cuota se paga al momento que se entrega el producto.
- Se aceptan todas las tarjetas de crédito pero tienen interés, 2 cuotas 10%, 3 cuotas 15%, 4 cuotas 20%, 5 cuotas 25%, 6 cuotas 30%, 9 cuotas 45% y hasta 12 cuotas 60%.
- Para terminar de confirmar la compra pedir nombre completo, dirección de entrega y teléfono de contacto.
- Las entregas suelen ser a los dos a cuatro días luego de la confirmación de la compra, por la siesta/tarde.
- Al terminar la conversación decir que nos volveremos a comunicar el día de la entrega para coordinar la misma.
- No generes información sensible, privada o especulativa.
- La misma pregunta debe generar la misma respuesta en cada interacción.
- Si desea comunicarse mediante una llamada decir que la atención es únicamente por whatsapp.

Tu objetivo es concretar la venta.
"""

# =========================
# ENDPOINTS PRODUCTOS
# =========================

@app.post("/products")
def add_products(products: List[ProductInput]):

    if not products:
        raise HTTPException(
            status_code=400,
            detail="La lista de productos está vacía"
        )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for product in products:

        if product.codigo in PRODUCT_DB:
            raise HTTPException(
                status_code=400,
                detail=f"El producto con código {product.codigo} ya existe, debes quitarlo de la lista."
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

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{product.codigo}_{i}")
            all_metadatas.append({"codigo": product.codigo})

        PRODUCT_DB[product.codigo] = product

    embeddings = embed_texts(all_chunks)

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings
    )

    return {
        "message": "Productos cargados correctamente",
        "total": len(products)
    }

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

# =========================
# RUN
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "challengefinal_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
