
import json
from pathlib import Path


def recall_at_k(
    expected_sources: list[str],
    retrieved_sources: list[str],
    k: int,
) -> float:
    """Calculate recall for one question."""
    expected = set(expected_sources)
    retrieved = set(retrieved_sources[:k])

    if not expected:
        return 0.0

    return len(expected & retrieved) / len(expected)


def evaluate(
    dataset_path: str,
    results_path: str,
    k_values: list[int] = [1, 3, 5, 10],
) -> None:
    """Compare retrieved sources with expected sources."""
    with open(dataset_path, encoding="utf-8") as file:
        dataset = json.load(file)

    with open(results_path, encoding="utf-8") as file:
        results = json.load(file)

    scores = {k: [] for k in k_values}

    for question, result in zip(
        dataset["questions"], results["questions"]
    ):
        expected = question["source"]
        retrieved = result["sources"]

        # Support either one source or a list of sources.
        if isinstance(expected, str):
            expected = [expected]

        for k in k_values:
            scores[k].append(
                recall_at_k(expected, retrieved, k)
            )

    print("Chunking evaluation")
    print("-" * 30)

    for k, values in scores.items():
        recall = sum(values) / len(values) if values else 0.0
        print(f"Recall@{k}: {recall:.3f}")


if __name__ == "__main__":
    evaluate(
        dataset_path="data/evaluation.json",
        results_path="data/search_results.json",
    )
