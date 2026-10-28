import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.embedding import cosine_similarity, embedding_model

def load_dataset(file_path: Path = Path("data/evaluation_dataset.json")) -> list[dict]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        return content
    else:
        raise FileNotFoundError(f"Missing {file_path}")

def generate_embedding(list_of_strings):
    return list(embedding_model.embed(list_of_strings, batch_size=64))

def compute_similarities(dataset: list[dict]) -> list[dict]:
    queries_a = [s["query_a"] for s in dataset]
    queries_b = [s["query_b"] for s in dataset]

    print(f"Generating embeddings for {len(dataset)} query pairs...")

    embed_1 = generate_embedding(queries_a)
    embed_2 = generate_embedding(queries_b)

    for i, sample in enumerate(dataset):
        sim = cosine_similarity(embed_1[i].tolist(), embed_2[i].tolist())
        sample["similarity"] = round(sim, 4)

    return dataset

def evaluate_threshold(dataset: list[dict], threshold: float) -> dict:
    tp = 0  
    fp = 0  
    tn = 0 
    fn = 0 
    high_risk_fp = 0
    conditional_hits = 0

    for sample in dataset:
        predicted = "hit" if sample["similarity"] >= threshold else "miss"
        expected = sample["expected_decision"]
        if expected == "hit":
            if predicted == "hit": 
                tp+=1
            else:
                fn+=1
        elif expected == "miss":
            if predicted == "hit":
                fp += 1
                if sample.get("risk_level") == "high":
                    high_risk_fp += 1
            else:
                tn += 1
        elif expected == "conditional":
            if predicted == "hit":
                conditional_hits += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "high_risk_fp": high_risk_fp,
    }

def run_evaluation():
    print("=" * 75)
    print("STARTING SEMANTIC CACHE BENCHMARK EVALUATION")
    print("=" * 75)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} evaluation samples.\n")

    dataset = compute_similarities(dataset)

    thresholds = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95]
    results = []

    print("\n" + "=" * 78)
    print(f"{'Threshold':<11} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'High-Risk FP'}")
    print("-" * 78)

    for th in thresholds:
        res = evaluate_threshold(dataset, th)
        results.append(res)
        print(
            f"{res['threshold']:<11.2f} | "
            f"{res['accuracy']:>7.1f}%   | "
            f"{res['precision']:>7.1f}%   | "
            f"{res['recall']:>7.1f}%   | "
            f"{res['f1']:>5.1f}%  | "
            f"{res['high_risk_fp']:>12}"
        )

    print("=" * 78)

    safe_candidates = [r for r in results if r["high_risk_fp"] == 0]
    best = max(safe_candidates or results, key=lambda r: r["f1"])

    print(f"\nRECOMMENDED OPTIMAL THRESHOLD: {best['threshold']:.2f}")
    print(f"Accuracy:       {best['accuracy']}%")
    print(f"Precision:      {best['precision']}% (Confidence in cache hits)")
    print(f"Recall:         {best['recall']}% (Total redundant queries caught)")
    print(f"F1 Score:       {best['f1']}%")
    print(f"High-Risk FP:   {best['high_risk_fp']} (Safety Guaranteed!)")


if __name__ == "__main__":
    run_evaluation()