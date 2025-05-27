from typing import List, Dict, Optional
from openai import AsyncOpenAI
import numpy as np
import logging
from app.services.cache_service import CacheService, cache
from app.services.redis_service import redis_client

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.logger = logging.getLogger(__name__)
        self.cache_service = CacheService()

    @cache(expire=3600)
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's API."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise

    @cache(expire=3600)
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise 