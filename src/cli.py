import json
import uuid
from pick_chunk import Indexer
from search import Retriever
from get_answer import Answer
from llm_sdk.llm_sdk import Small_LLM_Model
from src.evaluation import eval
from src.models import (MinimalAnswer,
                        StudentSearchResultsAndAnswer)


path = "data/raw/vllm-0.10.1"


class CLI:
    """Command line interface for indexing, searching and answering."""

    def __init__(self) -> None:
        """Load the small LLM model."""
        self.qwen = Small_LLM_Model()

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the chunk index from the raw data folder."""
        indexing = Indexer()
        indexing.build(path, max_chunk_size)

    def search(self, query: str, k: int = 10) -> None:
        """Print the top k chunk locations matching the query."""
        chunk = Retriever()
        outputs = chunk.piked_chunk(query, k)
        for output in outputs:
            print(
                f"{output['file_path']} "
                f"[{output['first_char']}:{output['last_char']}]")

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 5,
            save_directory: str = "data/search_results"
                        ) -> None:
        """Run the search on a whole dataset and save the results."""
        ret = Retriever()
        ret.save_result(dataset_path, k, save_directory)

    def answer(self, query: str, k: int = 10) -> None:
        """Retrieve k chunks for the query and print the generated answer."""
        retriev = Retriever()
        chunks = retriev.piked_chunk(query, k)
        question_id = str(uuid.uuid4())
        respond = Answer(self.qwen, chunks, query, question_id)
        print(respond.generate_answer())

    def answer_dataset(self,
                       student_search_result_path: str,
                       save_directory: str = "data/output",
                       k: int = 5
                       ) -> None:
        """Answer every question of a search result file and save them."""
        chunk = Retriever()

        with open(student_search_result_path, "r") as file:
            data = json.load(file)

        answers = data["search_results"]
        results = []
        for answer in answers:
            chunks = chunk.piked_chunk(answer["question"], k)
            retiv = Answer(
                self.qwen,
                chunks,
                answer["question"],
                answer["question_id"]
            )

            result = MinimalAnswer(
                question_id=answer["question_id"],
                question=answer["question"],
                retrieved_sources=answer["retrieved_sources"],
                answer=retiv.generate_answer()
            )
            results.append(result)

            output = StudentSearchResultsAndAnswer(
                search_results=results,
                k=k
            )

        s_dir = save_directory
        with open(s_dir + "/search_result_and_answer.json", "w") as file:
            json.dump(output.model_dump(), file, indent=4)

        print(f"The operation Done file saved in {s_dir}")

    def evaluate(
            self,
            student_search_result_path: str,
            dataset_path: str,
            k: int = 10) -> None:
        """Evaluate the search results against the dataset."""
        print("Starting evaluation...")

        eval(
            dataset_path,
            student_search_result_path,
            k
        )

        print("Evaluation End")
