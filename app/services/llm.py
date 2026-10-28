from openai import AsyncOpenAI
from app.core.config import settings
from time import perf_counter
api_key = settings.GROQ_API_KEY
base_url = settings.GROQ_BASE_URL

async def generate_completion(prompt: str, model: str | None = None, temperature: float = 0.7) -> tuple[str, str, float]:
    selected_model = model or settings.DEFAULT_MODEL
    client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
    start_time = perf_counter()
    response = await client.chat.completions.create(
            model= selected_model,
            messages = [
                {"role":"user", "content": prompt}
            ],
            temperature = temperature
        )
    end_time = perf_counter()
    latency_ms = (end_time - start_time) * 1000
    content = response.choices[0].message.content or ""
    return (content, selected_model, latency_ms)
