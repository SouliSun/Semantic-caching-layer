from pathlib import Path
import os

def load_env_file(dotenv_file: Path = Path(".env")) -> None:
    if not dotenv_file.is_file():
        return
    with open(dotenv_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            clean_value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), clean_value)

load_env_file()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")

    REDIS_URL: str = os.getenv("REDIS_URL", "")

    PROJECT_NAME: str = "Semantic Caching Layer"
    VERSION: str = "0.1.0"

settings = Settings()