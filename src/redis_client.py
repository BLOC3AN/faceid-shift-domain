import os
import logging
import numpy as np
import redis
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("FaceClustering.RedisClient")


class RedisCacheClient:
    """
    Reusable Redis client for caching embeddings and general key-value data.
    Reads connection config from .env (REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB).
    """

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Redis server."""
        try:
            redis_password = os.getenv("REDIS_PASSWORD", "") or None
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))

            self.client = redis.Redis(
                host=host,
                port=port,
                password=redis_password,
                db=db,
                socket_connect_timeout=3
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port} (db={db})")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self.client = None

    @property
    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        return self.client is not None

    def get(self, key: str) -> Optional[bytes]:
        """Get raw bytes from Redis by key."""
        if not self.client:
            return None
        try:
            return self.client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET failed for '{key}': {e}")
            return None

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set raw bytes in Redis. Optional TTL in seconds."""
        if not self.client:
            return False
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Redis SET failed for '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.warning(f"Redis DELETE failed for '{key}': {e}")
            return False

    # --- Embedding-specific helpers ---

    def save_embedding(self, key: str, embedding: np.ndarray) -> bool:
        """Serialize and store a numpy embedding vector in Redis."""
        try:
            data = embedding.astype(np.float32).tobytes()
            success = self.set(key, data)
            if success:
                logger.info(f"Embedding cached to Redis: {key}")
            return success
        except Exception as e:
            logger.warning(f"Failed to save embedding '{key}': {e}")
            return False

    def load_embedding(self, key: str) -> Optional[np.ndarray]:
        """Load and deserialize a numpy embedding vector from Redis."""
        data = self.get(key)
        if data is not None:
            embedding = np.frombuffer(data, dtype=np.float32).copy()
            logger.info(f"Embedding loaded from Redis cache: {key} (shape={embedding.shape})")
            return embedding
        return None
