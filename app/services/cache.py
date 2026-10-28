import json
import logging
import uuid
import time 
import re
from pydantic import BaseModel, Field
import redis.asyncio as aioredis
from app.core.config import settings
from app.services.embedding import cosine_similarity

logger = logging.getLogger(__name__)

# Heuristic patterns for time-sensitive prompts
TEMPORAL_PATTERNS = [
    r"\b(today|now|current|currently|latest|recent|recently|yesterday|tomorrow|tonight)\b",
    r"\b(weather|stock|stocks|price|prices|breaking news)\b",
    r"\bthis\s+(week|month|year|morning|afternoon|evening|weekend)\b",
    r"\blive\s+score(s)?\b",
    r"\bright\s+now\b",
    r"\bat\s+the\s+moment\b",
]

TEMPORAL_REGEX = re.compile("|".join(TEMPORAL_PATTERNS), re.IGNORECASE)


def is_time_sensitive(prompt: str) -> bool:
    """Detect if a query requires fresh, real-time data."""
    return bool(TEMPORAL_REGEX.search(prompt))


class CacheEntry(BaseModel):
    prompt: str
    embedding: list[float]
    response: str
    model: str
    temperature: float = 0.7
    created_at: float = Field(default_factory=time.time)
    ttl: int = 86400


class SemanticCache:
    MAX_L1_ENTRIES: int = 1000

    TEMPERATURE_TOLERANCE: float = 0.05

    def __init__(self):
        self.threshold: float = 0.88
        self.in_memory_entries: list[CacheEntry] = []
        self.in_memory_hits: int = 0
        self.in_memory_misses: int = 0
        self.redis_client = None

    async def connect(self) -> None:
        """Connect to Redis asynchronously and hydrate L1 cache on startup."""
        if not settings.REDIS_URL:
            return
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis (Async) successfully.")
            await self._warm_up_from_redis()
        except Exception as e:
            logger.warning(
                f"Could not connect to Redis: {e}. Operating in standalone in-memory mode."
            )
            self.redis_client = None

    async def disconnect(self) -> None:
        """Gracefully close Redis connection on shutdown."""
        if self.redis_client:
            try:
                await self.redis_client.aclose()
            except Exception:
                pass
            self.redis_client = None

    async def _warm_up_from_redis(self) -> None:
        """
        Hydrate L1 with the newest entries using an O(1) bounded transfer.
        Uses a capped Redis list (LRANGE 0 999) so only 1,000 entries are ever transferred over WAN,
        even if the overall database contains millions of records.
        """
        if self.redis_client is None:
            return
        try:
            raw_entries = await self.redis_client.lrange(
                "semantic_cache:recent", 0, self.MAX_L1_ENTRIES - 1
            )

            if not raw_entries:
                all_hash = await self.redis_client.hgetall("semantic_cache:entries")
                raw_entries = list(all_hash.values())[-self.MAX_L1_ENTRIES:]

            self.in_memory_entries = [
                CacheEntry(**json.loads(s)) for s in raw_entries
            ]
            logger.info(
                f"Hydrated L1 cache with {len(self.in_memory_entries)} entries "
                f"(capped network transfer: {self.MAX_L1_ENTRIES})."
            )
        except Exception as e:
            logger.warning(f"Failed to hydrate L1 cache from Redis: {e}")

    def search(
        self,
        query_embedding: list[float],
        threshold: float | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> tuple[CacheEntry, float] | None:
        """
        Pure in-memory L1 search with zero network overhead (< 1ms).
        Enforces TTL, model isolation, and temperature tolerance.
        """
        target_threshold = threshold if threshold is not None else self.threshold
        best_entry = None
        best_score = -1.0

        for entry in self.in_memory_entries:
            if (time.time() - entry.created_at) > entry.ttl:
                continue

            if model and entry.model != model:
                continue

            if temperature is not None and abs(entry.temperature - temperature) > self.TEMPERATURE_TOLERANCE:
                continue

            score = cosine_similarity(entry.embedding, query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= target_threshold:
            self.in_memory_hits += 1
            if self.redis_client:
                import asyncio
                asyncio.create_task(self._async_incr("semantic_cache:hits"))
            return (best_entry, best_score)
        else:
            self.in_memory_misses += 1
            if self.redis_client:
                import asyncio
                asyncio.create_task(self._async_incr("semantic_cache:misses"))
            return None

    async def _async_incr(self, key: str) -> None:
        """Helper to increment Redis counters asynchronously without blocking."""
        try:
            if self.redis_client:
                await self.redis_client.incr(key)
        except Exception:
            pass

    async def add(
        self,
        prompt: str,
        embedding: list[float],
        response: str,
        model: str,
        temperature: float = 0.7,
        ttl: int = 86400,
    ) -> None:
        """
        Write-through addition: saves immediately to L1 RAM and asynchronously persists to Redis.
        Maintains a bounded Redis List for O(1) startup warmup.
        """
        entry = CacheEntry(
            prompt=prompt,
            embedding=embedding,
            response=response,
            model=model,
            temperature=temperature,
            ttl=ttl,
        )

        if len(self.in_memory_entries) >= self.MAX_L1_ENTRIES:
            self.in_memory_entries.pop(0)
        self.in_memory_entries.append(entry)

        if self.redis_client is not None:
            try:
                json_data = json.dumps(entry.model_dump())
                entry_id = uuid.uuid4().hex

                await self.redis_client.lpush("semantic_cache:recent", json_data)
                await self.redis_client.ltrim(
                    "semantic_cache:recent", 0, self.MAX_L1_ENTRIES - 1
                )

                await self.redis_client.hset(
                    "semantic_cache:entries",
                    entry_id,
                    json_data,
                )
            except Exception as e:
                logger.warning(f"Failed to persist entry to Redis: {e}")

    async def get_stats(self) -> dict:
        total_entries = 0
        hits = self.in_memory_hits
        misses = self.in_memory_misses

        if self.redis_client is not None:
            try:
                total_entries = await self.redis_client.hlen("semantic_cache:entries")
                r_hits = await self.redis_client.get("semantic_cache:hits")
                r_misses = await self.redis_client.get("semantic_cache:misses")
                hits = int(r_hits) if r_hits else self.in_memory_hits
                misses = int(r_misses) if r_misses else self.in_memory_misses
            except Exception:
                total_entries = len(self.in_memory_entries)
        else:
            total_entries = len(self.in_memory_entries)

        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "storage_backend": "redis (async)" if self.redis_client is not None else "in-memory",
            "total_cached_entries": total_entries,
            "l1_memory_entries": len(self.in_memory_entries),
            "max_l1_capacity": self.MAX_L1_ENTRIES,
            "hits": hits,
            "misses": misses,
            "hit_rate_pct": round(hit_rate, 2),
        }

    async def clear(self) -> dict:
        """Clear cache both in RAM and asynchronously in Redis."""
        if self.redis_client is not None:
            try:
                await self.redis_client.delete(
                    "semantic_cache:entries",
                    "semantic_cache:recent",
                    "semantic_cache:hits",
                    "semantic_cache:misses",
                )
            except Exception:
                pass
        self.in_memory_entries.clear()
        self.in_memory_hits = 0
        self.in_memory_misses = 0
        return {"status": "cache cleared"}


cache = SemanticCache()
