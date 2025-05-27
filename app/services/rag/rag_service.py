from typing import List, Dict, Optional
from openai import AsyncOpenAI
import logging
from app.services.cache_service import CacheService, cache
from app.services.redis_service import redis_client
from app.services.rag.embedding.embedding_service import EmbeddingService
from app.services.rag.vector_db.chroma_service import ChromaService
from app.services.rag.document.document_service import DocumentService

class RAGService:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.logger = logging.getLogger(__name__)
        self.cache_service = CacheService()
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaService()
        self.document_service = DocumentService()

    @cache(expire=3600)
    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Search in vector store
            results = await self.chroma_service.search(query_embedding, top_k=top_k)
            
            # Sort by relevance score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise

    @cache(expire=3600)
    async def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a response using retrieved context."""
        try:
            # Format context with source information
            context_text = "\n\n".join([
                f"Source {i+1} ({doc['metadata']['title']}, Chunk {doc['metadata']['chunk_index']+1}/{doc['metadata']['total_chunks']}):\n{doc['content']}"
                for i, doc in enumerate(context)
            ])
            
            # Generate response
            prompt = f"""You are a medical knowledge assistant. Based on the following medical guidelines and clinical examples, please answer the question.
            Focus on providing evidence-based information from the provided context.
            If the context doesn't contain relevant information, say so.
            
            Structure your response to include:
            1. Direct answer to the question
            2. Key points from the guidelines
            3. Any relevant clinical examples
            4. Source citations in the format: [Source: Document Title, Chunk X/Y]
            
            Context:
            {context_text}
            
            Question: {query}
            
            Answer:"""
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    @cache(expire=3600)
    async def process_query(self, query: str, top_k: int = 3) -> Dict:
        """Process a query using RAG pipeline."""
        try:
            # Retrieve relevant documents
            docs = await self.retrieve(query, top_k=top_k)
            
            # Generate response
            response = await self.generate_response(query, docs)
            
            return {
                "query": query,
                "response": response,
                "sources": docs
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

    @cache(expire=3600)
    async def answer_question(self, question: str, documents: List[Dict]) -> Dict:
        """Answer a question using RAG on the provided documents."""
        try:
            # Prepare documents (chunking and cleaning)
            prepared_docs = await self.document_service.prepare_documents(documents)
            
            # Store documents in vector store
            await self.chroma_service.add_documents(prepared_docs)
            
            # Process the question
            result = await self.process_query(question)
            
            if not result["sources"]:
                return {
                    "answer": "The context provided does not contain information related to your question.",
                    "sources": []
                }
            
            return {
                "answer": result["response"],
                "sources": result["sources"]
            }
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            raise 