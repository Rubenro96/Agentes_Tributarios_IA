from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import sqlite3
import uuid
import os
from agents import run_conversation

# Inicialización de la aplicación FastAPI
app = FastAPI(title="Rubén Rubio AI Assistant API")

# Configuración CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de la base de datos
DB_PATH = "./data/sqlite/conversation_history.db"

# Definición de modelos Pydantic
class Message(BaseModel):
    thread_id: str
    pregunta: str
    revision: Optional[str] = None
    timestamp: str

class ResponseRequest(BaseModel):
    thread_id: Optional[str]
    message: str

class AIResponse(BaseModel):
    response: Any

# Función para asegurar que la base de datos existe
def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            pregunta TEXT NOT NULL,
            revision TEXT,
            timestamp TEXT NOT NULL
        )
        ''')
        conn.commit()
        conn.close()

# Función para obtener conexión a la base de datos
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
    try:
        return conn
    finally:
        conn.close()

# Endpoints de la API
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API del Asistente de Rubén Rubio"}

@app.get("/conversations/{thread_id}", response_model=List[dict])
def get_conversation(thread_id: str):
    """Obtiene el historial de una conversación específica."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pregunta, revision, timestamp FROM conversation_history WHERE thread_id = ? ORDER BY timestamp ASC", 
            (thread_id,)
        )
        conversation = [dict(row) for row in cursor.fetchall()]
        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener la conversación: {str(e)}")
    finally:
        conn.close()


@app.get("/recent-conversations", response_model=List[Tuple[str, str]])
def get_recent_conversations():
    """Obtiene una lista de las conversaciones recientes."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        # Obtiene el primer mensaje de cada thread_id ordenado por timestamp descendente
        cursor.execute("""
            SELECT c1.thread_id, c1.pregunta 
            FROM conversation_history c1
            JOIN (
                SELECT thread_id, MIN(timestamp) as first_timestamp
                FROM conversation_history
                GROUP BY thread_id
            ) c2 ON c1.thread_id = c2.thread_id AND c1.timestamp = c2.first_timestamp
            ORDER BY c1.timestamp DESC
        """)
        # Convertimos a lista de tuplas (thread_id, pregunta)
        conversations = [(row[0], row[1]) for row in cursor.fetchall()]
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener conversaciones recientes: {str(e)}")
    finally:
        conn.close()

@app.post("/generate-response", response_model=AIResponse)
async def generate_ai_response(request: ResponseRequest):
    """
    Genera una respuesta usando un sistema de agentes.
    """
    try:
        result = run_conversation(request.message, thread_id=request.thread_id)
        # Extract just the revision field or another specific field you want to return
        return AIResponse(response=result["revision"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {str(e)}")

# Inicializar la base de datos al arrancar
@app.on_event("startup")
def startup_db_client():
    init_db()

# Si ejecutamos directamente este archivo
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)