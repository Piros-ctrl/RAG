import os
import json
import bm25s
from typing import List, Dict, Any
from pathlib import Path
from src.models import (RagDataset,
                        MinimalSource,
                        UnansweredQuestion,
                        MinimalSearchResults,
                        StudentSearchResults)


class Retriever:
    """Load the BM25 index and retrieve chunks for questions."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Load the saved BM25 index, or fail if it does not exist."""
        index_path = Path(processed_dir) / "bm25_index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. Run 'index' first."
            )
        self.bm25 = bm25s.BM25.load(str(index_path), load_corpus=True)
        self.num_chunks = self.bm25.scores["num_docs"]

    def piked_chunk(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return the top k chunks that best match the query."""
        k = min(k, self.num_chunks)

        query_tokens = bm25s.tokenize(query)
        if len(query_tokens.ids[0]) == 0:
            return []

        docs, _scores = self.bm25.retrieve(query_tokens, k=k)
        return list(docs[0])

    def creat_dataset(self, dataset: str, k: int) -> Dict[str, Any]:
        """Retrieve k sources for every dataset question and return them."""
        search_result = []
        with open(dataset, "r") as p_file:
            data = RagDataset.model_validate_json(p_file.read())

        for data in data.rag_questions:
            question_obj = UnansweredQuestion(
                question_id=data.question_id,
                question=data.question
            )
            question = question_obj.model_dump()
            outputs = self.piked_chunk(data.question, k)
            retrieved_sources = []

            for output in outputs:
                source = MinimalSource(
                    file_path=output["file_path"],
                    first_character_index=output["first_char"],
                    last_character_index=output["last_char"]
                )
                retrieved_sources.append(source)

            resutl = MinimalSearchResults(
                question_id=question["question_id"],
                question=question["question"],
                retrieved_sources=retrieved_sources
            )

            search_result.append(resutl)
        output = StudentSearchResults(
            search_results=search_result,
            k=k
        )
        return output.model_dump()

    def save_result(self, dataset: str, k: int, s_dir: str) -> None:
        """Run the search on the dataset and save the results as JSON."""
        os.makedirs(s_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(dataset))[0]
        if not name.endswith(".json"):
            name += ".json"
        output_file = os.path.join(s_dir, name)

        with open(output_file, "w") as file:
            json.dump(self.creat_dataset(dataset, k), file, indent=4)

        print(f"The operation Done file saved in {output_file}")
