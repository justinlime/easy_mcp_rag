import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

logger = logging.getLogger("easy_mcp_rag.vectorstore")

class VectorStore:
    def __init__(self, host: str, port: int, model_name: str, device: str = "cpu"):
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.device = device
        
        logger.info(f"Connected to Qdrant at {host}:{port}")
        logger.info(f"Loaded embedding model: {model_name} on device: {device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def create_collection(self, collection_name: str, force: bool = False):
        if force and self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
    
    def index_documents(self, collection_name: str, documents: List[Dict[str, Any]], 
                       batch_size: int = 32):
        points = []
        
        for i, doc in enumerate(documents):
            content = doc["content"]
            embedding = self.model.encode(content, show_progress_bar=False)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "content": content,
                    **doc.get("metadata", {})
                }
            )
            points.append(point)
            
            if len(points) >= batch_size:
                self.client.upsert(collection_name=collection_name, points=points)
                logger.debug(f"Indexed batch of {len(points)} documents")
                points = []
        
        if points:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Indexed final batch of {len(points)} documents")
        
        logger.info(f"Indexed {len(documents)} documents into {collection_name}")
    
    def search(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.model.encode(query, show_progress_bar=False)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return [
            {
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            }
            for hit in results
        ]
