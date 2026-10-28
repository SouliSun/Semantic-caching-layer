from fastembed import TextEmbedding
import numpy as np

embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

def get_embedding(text: str) -> list[float]:
    embedding = embedding_model.embed([text])
    result = list(embedding)[0].tolist()
    return result


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate the cosine similarity between two embedding vectors."""
    if len(v1) != len(v2):
        return 0.0
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

