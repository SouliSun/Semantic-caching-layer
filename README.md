# Semantic Caching Layer for LLM APIs

A high-performance middleware caching gateway that sits between client applications and LLM providers (e.g., Groq, OpenAI). It detects semantically equivalent queries using local vector embeddings, serving cached responses in **< 15ms** at **$0 cost**.

---

## Architecture

```
Client Request
      │
      ▼
FastAPI Gateway (Non-blocking async)
      │
      ├──► [1. Heuristic Freshness & Temp Guard] ──► (Bypass if time-sensitive or temp > 0.7)
      │
      ├──► [2. Threadpool Vector Embedding] ──────► (fastembed on CPU)
      │
      └──► [3. Two-Tier Cache Lookup]
                 │
                 ├── L1: In-Memory RAM (< 1ms, 0 network I/O)
                 └── L2: Upstash Cloud Redis (Async write-through persistence)
                 │
           ┌─────┴─────────────────────┐
           ▼                           ▼
      [CACHE HIT]                 [CACHE MISS]
   Return instantly (<15ms)      Call Groq API (~800ms)
   Cost: $0.00                   Persist to L1 & L2 in background
```

---

## Key Features

- **Two-Tier (L1/L2) Storage**: Sub-millisecond reads from local RAM; durable write-through cloud persistence in Redis.
- **O(1) Bounded Startup Hydration**: Uses a capped Redis list (`LRANGE 0 999`) to prevent network spikes at boot.
- **Model & Temperature Isolation**: Guarantees requests for high-tier models or creative temperatures never cross-contaminate.
- **Freshness & Temporal Safety**: Phrase-aware regex excludes time-sensitive questions (*"today"*, *"this week"*, *"stocks"*) from reads and writes.
- **Client Cache-Control**: Supports explicit `no_cache: true` overrides.
- **Fully Asynchronous**: Decouples CPU embedding (`asyncio.to_thread`) and Redis persistence (`BackgroundTasks`).

---

## Benchmark Results (400 Adversarial Pairs)

Evaluated against a 400-sample pairwise dataset including paraphrases and lexical hard negatives:

| Threshold | Accuracy | Precision | Recall | High-Risk FP | Notes |
|:---:|:---:|:---:|:---:|:---:|---|
| **0.80** | 60.4% | 53.1% | **87.3%** | 10 | Aggressive cost savings |
| **0.88** | **66.8%** | **64.8%** | 54.2% | 5 | Balanced default |
| **0.95** | 57.5% | 66.7% | 7.2% | **0** | Zero false-positive safety |

---

## Quick Start

### 1. Installation
```powershell
# Clone repo & navigate
git clone https://github.com/SouliSun/SemanticCachingLayer.git
cd SemanticCachingLayer

# Activate venv & install
pip install fastapi uvicorn openai fastembed redis pydantic
```

### 2. Configure Environment (`.env`)
```env
GROQ_API_KEY="your-groq-api-key"
DEFAULT_MODEL="openai/gpt-oss-20b"
REDIS_URL="rediss://default:password@endpoint.upstash.io:6379"
```

### 3. Run the Server
```powershell
uvicorn app.api.main:app --reload
```
Interactive API documentation available at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

- **`POST /api/v1/chat`**: Query endpoint with automatic semantic caching.
- **`GET /api/v1/cache/stats`**: Live metrics (hits, misses, hit rate %, L1 RAM count).
- **`DELETE /api/v1/cache`**: Flush local and cloud cache.
- **`GET /api/v1/health`**: Health status check.
