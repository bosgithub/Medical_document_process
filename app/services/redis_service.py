import os
import redis.asyncio as redis
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True
    )
    logger.info("Successfully initialized Redis client")
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {str(e)}")
    redis_client = None 