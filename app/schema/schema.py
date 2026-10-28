from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input text or question")
    model: str | None = Field(
        default=None,
        examples=["openai/gpt-oss-20b"],
        description="Override the default model from config"
    ) 
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Control Randomness (0.0 = deterministic, 2.0 = very random)."
    )
    no_cache: bool = Field(
        default=False,
        description="Set to true to bypass semantic cache and force fresh LLM generation"
    )

class ChatResponse(BaseModel):
    response: str = Field(description="The text returned by the LLM")
    model: str = Field(description="The model that processed the response")
    cached: bool = Field(
        default=False,
        description="Flag indicating whether the response was a cache hit or a miss"
    )
    latency_ms: float = Field(description="Request duration in milliseconds")
    similarity_score: float | None = Field(
        default=None,
        description="Cosine similarity score for cache hits"
    )
