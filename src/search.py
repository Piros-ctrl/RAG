# import json
import pickle
import json, os
from pathlib import Path
# from typing import List
# from tqdm import tqdm
from src.pick_chunk import tokenize
from src.models import (RagDataset,
                        MinimalSource,
                        UnansweredQuestion,
                        MinimalSearchResults,
                        StudentSearchResults)


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

    def piked_chunk(self, query, k):
        tokenized_query = tokenize(query)
        chunk = [chunk for chunk in self.chunks]
        return self.bm25.get_top_n(tokenized_query, chunk, n=k)

    def creat_dataset(self, dataset, k):
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

    def save_result(self,dataset, k, s_dir):
        os.makedirs(s_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(dataset))[0]
        output_file = os.path.join(s_dir, "search_results.json")

        with open(output_file, "w") as file:
            json.dump(self.creat_dataset(dataset, k), file, indent=4)
