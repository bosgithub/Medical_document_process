from typing import List, Dict
import re
import logging
from app.services.cache_service import CacheService, cache

class DocumentService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_service = CacheService()
        self.chunk_size = 800
        self.chunk_overlap = 150

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters but keep medical terms
            text = re.sub(r'[^\w\s\-.,;:()/]', '', text)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            raise

    def split_into_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        try:
            chunk_size = chunk_size or self.chunk_size
            overlap = overlap or self.chunk_overlap
            
            # Clean the text first
            text = self.clean_text(text)
            
            # Split into sentences to avoid breaking mid-sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting text into chunks: {str(e)}")
            raise

    @cache(expire=3600)
    async def prepare_document(self, document: Dict) -> List[Dict]:
        """Prepare a document for vector storage."""
        try:
            # Split document into chunks
            chunks = self.split_into_chunks(document["content"])
            
            # Create chunk documents with metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "id": f"{document['id']}_chunk_{i}",
                    "content": chunk,
                    "metadata": {
                        "title": document["title"],
                        "document_id": document["id"],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                chunk_docs.append(chunk_doc)
            
            return chunk_docs
        except Exception as e:
            self.logger.error(f"Error preparing document: {str(e)}")
            raise

    @cache(expire=3600)
    async def prepare_documents(self, documents: List[Dict]) -> List[Dict]:
        """Prepare multiple documents for vector storage."""
        try:
            all_chunks = []
            for doc in documents:
                chunk_docs = await self.prepare_document(doc)
                all_chunks.extend(chunk_docs)
            return all_chunks
        except Exception as e:
            self.logger.error(f"Error preparing documents: {str(e)}")
            raise 