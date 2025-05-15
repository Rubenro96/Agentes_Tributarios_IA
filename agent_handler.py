import sqlite3
import json
import datetime
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class ConversationHistoryManager:
    """Permite registrar y recuperar la historia de conversaciones de los usuarios."""

    def __init__(self, db_name="conversation_history.db"):
        # Crear la carpeta data/sqlite si no existe
        data_dir = Path("agent_system/data/sqlite")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruta completa a la base de datos
        db_path = data_dir / db_name
        
        # Crear/conectar a la base de datos
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()
        
        # Configurar la base de datos
        self._setup_db()
        
        print(f"Conversation history database created at: {db_path}")

    def _setup_db(self):
        """Crea la tabla conversation_history si no existe."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            pregunta TEXT NOT NULL,
            tipo TEXT NOT NULL,
            contenido TEXT NOT NULL,
            consulta_RAG TEXT,
            plan TEXT,
            contexto TEXT,
            respuesta TEXT,
            revision TEXT,
            timestamp TEXT NOT NULL
        )
        ''')
        
        # Crear un índice para mejorar el rendimiento de búsqueda por thread_id
        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_thread_id ON conversation_history(thread_id)
        ''')
        
        self.conn.commit()

    def log_interaction(self, thread_id: str, pregunta: str, tipo: str, contenido: str, consulta_RAG: str, plan: str, contexto: str, respuesta: str, revision: str) -> None:
        """
        Realiza un log de cada interacción del usuario con el sistema de agentes.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        self.cursor.execute(
            "INSERT INTO conversation_history (thread_id, pregunta, tipo, contenido, consulta_RAG, plan, contexto, respuesta, revision, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, pregunta, tipo, contenido, consulta_RAG, plan, contexto, respuesta, revision, timestamp)
        )
        
        self.conn.commit()
        
        print(f"Logged interaction for thread: {thread_id}")

    def get_conversation_history(self, thread_id: str) -> List[Dict[str, str]]:
        """
        Recupera registros a partir del id del hilo.
        
        Args:
            thread_id: Identificador único del hilo.
            
        Returns:
            Los registros de conversaciones anteriores.
        """
        self.cursor.execute(
            "SELECT * FROM conversation_history WHERE thread_id = ? ORDER BY timestamp",
            (thread_id,)
        )
        
        history = []
        for row in self.cursor.fetchall():
            history.append({
                "pregunta": row[1],
                "tipo": row[2],
                "contenido": row[3],
                "consulta_RAG": row[4],
                "plan": row[5],
                "contexto": row[6],
                "respuesta": row[7],
                "revision": row[8],
                "timestamp": row[9]
            })

        print(f"Retrieved {len(history)} interactions for thread: {thread_id}")
        return history

    def delete_thread_history(self, thread_id: str) -> int:
        """
        Borra la conversació entera para un posible hilo.
        
        Args:
            thread_id: Identificador único del hilo.
            
        Returns:
            Número de registros eliminados.
        """
        self.cursor.execute(
            "DELETE FROM conversation_history WHERE thread_id = ?",
            (thread_id,)
        )
        
        deleted_count = self.cursor.rowcount
        self.conn.commit()
        
        print(f"Deleted {deleted_count} interactions for thread: {thread_id}")
        return deleted_count

    def get_all_thread_ids(self) -> List[str]:
        """
        Retrieves all unique thread IDs in the database.
        
        Returns:
            List of thread IDs
        """
        self.cursor.execute(
            "SELECT DISTINCT thread_id FROM conversation_history"
        )
        
        thread_ids = [row[0] for row in self.cursor.fetchall()]
        return thread_ids

    def close(self):
        """Close the database connection properly"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def __del__(self):
        """Ensure the database connection is closed when the object is deleted"""
        self.close()