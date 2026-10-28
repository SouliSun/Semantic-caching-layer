from fastapi import APIRouter, BackgroundTasks
from app.core.config import settings
from app.schema.schema import ChatRequest, ChatResponse
from app.services.llm import generate_completion
from app.services.embedding import get_embedding
from app.services.cache import cache, is_time_sensitive
from time import perf_counter
import asyncio

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    start_time = perf_counter()

    query_embedding = await asyncio.to_thread(get_embedding, request.prompt)

    target_model = request.model or settings.DEFAULT_MODEL
    is_temporal = is_time_sensitive(request.prompt)

    is_cacheable = (not request.no_cache) and (not is_temporal) and (request.temperature <= 0.7)

    if is_cacheable:
        match = cache.search(
            query_embedding,
            model=target_model,
            temperature=request.temperature
        )
    else:
        match = None

    if match is not None:
        cached_entry, similarity_score = match
        latency_ms = (perf_counter() - start_time) * 1000
        return ChatResponse(
            response=cached_entry.response,
            model=cached_entry.model,
            cached=True,
            latency_ms=latency_ms,
            similarity_score=round(similarity_score, 4),
        )

    content, model, _ = await generate_completion(
        prompt=request.prompt, 
        model=request.model, 
        temperature=request.temperature
    )

    if is_cacheable:
        background_tasks.add_task(
            cache.add,
            prompt=request.prompt,
            embedding=query_embedding,
            response=content,
            model=model,
            temperature=request.temperature,
            ttl=86400
        )

    latency_ms = (perf_counter() - start_time) * 1000
    return ChatResponse(
        response=content,
        model=model,
        cached=False,
        latency_ms=latency_ms,
    )

@router.get("/cache/stats")
async def get_cache_stats():
    return await cache.get_stats()

@router.delete("/cache")
async def clear_cache():
    return await cache.clear()
