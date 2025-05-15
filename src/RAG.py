import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
from typing import List, Dict, Union
import uuid
from enum import Enum
from datetime import datetime
import os
import torch
import re
from huggingface_hub import login, snapshot_download

class DocumentType(Enum):
    GENERAL = "Normativa general aplicable"
    ITPAJD = "Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados" 
    ISD = "Impuesto sobre Sucesiones y Donaciones"
    IP = "Impuesto sobre el Patrimonio"
    JUEGO = "Tributos sobre el Juego"
    ANDALUCIA = "Comunidad Autónoma de Andalucía"
    ARAGON = "Comunidad Autónoma de Aragón"
    ASTURIAS = "Comunidad Autónoma del Principado de Asturias"
    BALEARES = "Comunidad Autónoma de las Islas Baleares"
    CANARIAS = "Comunidad Autónoma de Canarias"
    CANTABRIA = "Comunidad Autónoma de Cantabria"
    CASTILLA_LA_MANCHA = "Comunidad Autónoma de Castilla-La Mancha"
    CASTILLA_Y_LEON = "Comunidad Autónoma de Castilla y León"
    CATALUÑA = "Comunidad Autónoma de Cataluña"
    COMUNIDAD_VALENCIANA = "Comunidad Autónoma de la Comunidad Valenciana"
    EXTREMADURA = "Comunidad Autónoma de Extremadura"
    GALICIA = "Comunidad Autónoma de Galicia"
    LA_RIOJA = "Comunidad Autónoma de La Rioja"
    MURCIA = "Comunidad Autónoma de la Región de Murcia"
    MADRID = "Comunidad Autónoma de Madrid"
    NAVARRA = "Comunidad Foral de Navarra"
    PAIS_VASCO = "Comunidad Autónoma del País Vasco"
    

class VectorEmbeddings:
    def __init__(self, collection_name: str):
        #Modelo para embeddings
        self.model = SentenceTransformer(snapshot_download(repo_id="littlejohn-ai/bge-m3-spa-law-qa", local_dir="./models/BGE"), trust_remote_code=True)
        #Modelo para rerankear
        self.reranker = CrossEncoder(snapshot_download(repo_id="BAAI/bge-reranker-v2-m3", local_dir="./models/Reranker"))
        self.device = self.model.device
        print(f"Using device: {self.device}")
        self.name = collection_name
        if not os.path.exists("./data/chroma"):
            os.makedirs("./data/chroma")
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma/vec")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def move_to_gpu(self):
        if torch.cuda.is_available():
            self.model.to("cuda")
            print(f"Moved model to {self.model.device}") 
        else:
            print("GPU not available, model remains on CPU")
            
            
            
    def extract_MD(self, source: str) -> str:
        """
        Extrae contenido Markdown desde una fuente utilizando la librería Docling.
        
        Args:
            source: La fuente desde la cual se extraerá el contenido Markdown (normalmente una URL o una ruta local)
        """
        try:
            converter = DocumentConverter()
            md_content = converter.convert(source).document.export_to_markdown()
            return md_content
        except Exception as e:
            print(f"Error extracting Markdown: {str(e)}")
            raise


    def split_markdown_BOE(self, md_text: str, base_metadata: Dict) -> List[Dict]:
        """
        Divide el texto Markdown en chunks manejables con metadata conforme a la estructura de las leyes del BOE (Boletín Oficial del Estado)

        Args:
            md_text: Texto en formato Markdown
            base_metadata: Metadata base que se incluirá en cada chunk
        """
        try:
            # Encontrar todos los bloques que coinciden con la expresión regular
            blocks = re.findall(r'\[Bloque \d+: #[^\]]+\]', md_text)

            # Dividir el texto en base a los bloques encontrados
            split_texts = re.split(r'\[Bloque \d+: #[^\]]+\]', md_text)

            # Eliminar el primer elemento si está vacío (puede ocurrir si el texto comienza con un bloque)
            if split_texts[0] == '':
                split_texts = split_texts[1:]

            # Crear una lista de diccionarios con el contenido y el bloque correspondiente
            header_splits = [{"content": text, "block": block} for text, block in zip(split_texts, blocks)]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )

            final_chunks = []
            for split in header_splits:
                # Combinar metadata del bloque con base_metadata
                chunk_metadata = {**base_metadata, "block": split["block"]}

                if len(split["content"]) > 2000:
                    smaller_chunks = text_splitter.split_text(split["content"])
                    for i, chunk in enumerate(smaller_chunks):
                        final_chunks.append({
                            "content": chunk,
                            "metadata": {
                                **chunk_metadata,
                                "chunk_index": i,
                                "total_chunks": len(smaller_chunks)
                            }
                        })
                else:
                    final_chunks.append({
                        "content": split["content"],
                        "metadata": {
                            **chunk_metadata,
                            "chunk_index": 0,
                            "total_chunks": 1
                        }
                    })

            return final_chunks
        except Exception as e:
            print(f"Error splitting markdown: {str(e)}")
            raise e
        

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para una lista de textos."""
        try:
            embeddings = []
            for text in texts:
                embedding = self.model.encode(text)
                embeddings.append(embedding.tolist())
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    def process_document(self, 
                        source: str,
                        document_type: DocumentType,
                        document_id: str = None,
                        metadata: Dict = None,
                        related_docs: List[str] = None) -> str:
        """
        Procesa un documento con metadata enriquecida.

        Args:
            source: URL o ruta del documento
            document_type: Tipo de documento (GENERAL, SPECIFIC, etc.)
            document_id: ID único para el documento
            metadata: Metadata adicional del documento
            related_docs: Lista de IDs de documentos relacionados

        Returns:
            str: ID del documento procesado
        """
        try:
            # Generar document_id si no se proporciona
            if document_id is None:
                document_id = str(uuid.uuid4())

            # Metadata base del documento
            base_metadata = {
                "document_id": document_id,
                "document_type": document_type.value,
                "processing_date": datetime.now().isoformat(),
                "related_documents": ",".join(related_docs) if related_docs else ""
            }

            # Combinar con metadata adicional si existe
            if metadata:
                base_metadata.update(metadata)

            # Extraer y procesar contenido
            md_content = self.extract_MD(source)
            if not md_content:
                raise ValueError("No Markdown content was extracted")

            # Dividir en chunks con la metadata enriquecida
            chunks = self.split_markdown_BOE(md_content, base_metadata)
            if not chunks:
                raise ValueError("No chunks were generated from the document")

            # Preparar datos para ChromaDB
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [f"{document_id}-chunk-{i}" for i in range(len(chunks))]

            # Generar embeddings
            embeddings = self.get_embeddings(texts)

            # Añadir a la colección
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Successfully processed document {document_id} with {len(chunks)} chunks")
            return document_id

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise

    def query_similar(self, 
                     query_text: str, 
                     score: float = 0.6,
                     document_types: List[DocumentType] = None,
                     document_ids: List[str] = None,
                     include_related: bool = False) -> List[Dict]:
        """
        Búsqueda avanzada de documentos similares.

        Args:
            query_text: Texto de la consulta
            score: Mínimo de similitud del rerank score para que lo incluya en el contexto
            document_types: Tipos de documentos a incluir en la búsqueda
            document_ids: IDs específicos de documentos
            include_related: Si se incluyen documentos relacionados
        """
        try:
            # 1. Generar embeddings a partir de la consulta para la búsqueda vectorial
            query_embedding = self.model.encode(query_text).tolist()

            # 2.Construir filtro
            conditions = []
            
            if document_types:
                conditions.append({"document_type": {"$in": [dt.value for dt in document_types]}})
            
            if document_ids:
                if include_related:
                    related_docs = self.get_related_documents(document_ids)
                    document_ids.extend(related_docs)
                conditions.append({"document_id": {"$in": document_ids}})
            
            where = None
            if conditions:
                if len(conditions) == 1:
                    where = conditions[0]
                else:
                    where = {"$and": conditions}

            #3. Consulta inicial en Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=50,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # 4. Formatear los resultados
            formatted_results = self._format_results(results)
            
            # 5. Re-rankear los resultados utilizando el modelo CrossEncoder
            reranked_results = self._rerank_results(query_text, formatted_results, score)
            
            return reranked_results

        except Exception as e:
            print(f"Error querying similar documents: {str(e)}")
            return reranked_results

    def get_related_documents(self, document_ids: List[str]) -> List[str]:
        """Obtiene todos los documentos relacionados con los IDs proporcionados."""
        related_docs = set()
        try:
            # Obtener metadata de los documentos especificados
            where = {"document_id": {"$in": document_ids}}
            results = self.collection.get(where=where)

            # Extraer IDs de documentos relacionados
            for metadata in results['metadatas']:
                if 'related_documents' in metadata and metadata['related_documents']:
                    doc_list = metadata['related_documents'].split(',')
                    related_docs.update(doc_list)

            return list(related_docs)
        except Exception as e:
            print(f"Error getting related documents: {str(e)}")
            raise

    def _format_results(self, results: Dict) -> List[Dict]:
        """Formatea los resultados de la búsqueda."""
        formatted_results = []
        for i in range(len(results['documents'][0])):
            doc_content = results['documents'][0][i]
            
            #Si el contenido es una lista, lo convertimos a string
            if isinstance(doc_content, list):
                doc_content = ' '.join(doc_content)
            formatted_results.append({
                'document': doc_content,
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        return formatted_results
    
    def _rerank_results(self, query_text: str, results: List[Dict], score: float=0.6) -> List[Dict]:
        """
        Reordena los resultados usando un modelo CrossEncoder (self.reranker).
        Cada resultado es un diccionario con 
        {
            'document': ...,
            'metadata': ...,
            'distance': ...
        }
        """
        if not results:
            return results
        
        #Crear los pares (texto consulta - documento) para el modelo CrossEncoder
        pairs = [(query_text, r['document']) for r in results]
        
        #Obtener las puntuaciones de proximidad
        scores = self.reranker.predict(pairs)
        
        # Asociar las puntuaciones a los resultados y reordenarlos
        for i,r in enumerate(results):
            r["rerank_score"]=float(scores[i])
        
        #Ordenar los resultados por relevancia descendente
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        #Se queda con los resultados que hayan obtenido un rerank_score superior a la variable "score" que se introduce cuando se invoca al método
        selected_results = [r for r in results if r["rerank_score"] > score]

        return selected_results
    
