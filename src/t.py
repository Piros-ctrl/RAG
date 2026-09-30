import json
import pickle
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.indexer import tokenize
from src.models import MinimalSearchResults, MinimalSource, StudentSearchResults


class Retriever:
    def __init__(self, processed_dir: str = "data/processed") -> None:
        index_path = Path(processed_dir) / "bm25_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. Run 'index' first."
            )
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return top-k MinimalSource results for a single query."""
        if not query or not query.strip() or k <= 0:
            return []

        scores = self.bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            MinimalSource(
                file_path=self.chunks[i]["file_path"],
                first_character_index=self.chunks[i]["first_character_index"],
                last_character_index=self.chunks[i]["last_character_index"],
            )
            for i in top_idx
            if scores[i] > 0  # skip zero-score junk when query has no lexical overlap
        ]

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole JSON dataset, write StudentSearchResults JSON."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        questions = raw.get("rag_questions", [])
        results: List[MinimalSearchResults] = []

        for q in tqdm(questions, desc="Searching dataset"):
            sources = self.search(q["question"], k)
            results.append(
                MinimalSearchResults(
                    question_id=q["question_id"],
                    question=q["question"],
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=results, k=k)

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(dataset_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_path}")