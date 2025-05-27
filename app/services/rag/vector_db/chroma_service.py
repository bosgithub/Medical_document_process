from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
import logging
from app.services.cache_service import CacheService, cache
from app.services.redis_service import redis_client
import os
import json

class ChromaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_service = CacheService()
        
        try:
            # Set up persistent storage in the app directory
            persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'chroma_db')
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"
            )
            
            # Create or get collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name="medical_documents",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                self.logger.info("Successfully initialized ChromaDB collection")
            except Exception as e:
                self.logger.error(f"Error creating ChromaDB collection: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    @cache(expire=3600)
    async def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store."""
        try:
            # Prepare documents for storage
            ids = [doc["id"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [
                {
                    "title": doc["metadata"]["title"],
                    "document_id": doc["metadata"]["document_id"],
                    "chunk_index": doc["metadata"]["chunk_index"],
                    "total_chunks": doc["metadata"]["total_chunks"]
                }
                for doc in documents
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise

    @cache(expire=3600)
    async def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for similar documents using query embedding."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                documents.append({
                    "content": doc,
                    "metadata": metadata,
                    "score": 1 - distance  # Convert distance to similarity score
                })
            
            return documents
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            raise

    @cache(expire=3600)
    async def search_with_filters(
        self,
        query_embedding: List[float],
        filters: Dict,
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """Search for similar documents with additional filters."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                documents.append({
                    "content": doc,
                    "metadata": metadata,
                    "score": 1 - distance  # Convert distance to similarity score
                })
            
            return documents
        except Exception as e:
            self.logger.error(f"Error searching documents with filters: {str(e)}")
            raise 