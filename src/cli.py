import json, uuid
from pick_chunk import Indexer
from search import Retriever
from get_answer import Answer
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import (MinimalAnswer,
                        StudentSearchResultsAndAnswer)


path = "data/raw/vllm-0.10.1"
class CLI:
    def __init__(self):
        self.qwen = Small_LLM_Model()

    def index(self, max_chunk_size: int = 2000):
        indexing = Indexer()
        indexing.build(path, max_chunk_size)

    def search(self, query: str, k: int = 10):
        chunk = Retriever()
        outputs = chunk.piked_chunk(query, k)
        for output in outputs:
            print(
                f"{output['file_path']} "
                f"[{output['first_char']}:{output['last_char']}]")

    def search_dataset(self,
                        dataset_path: str,
                        k: int = 10,
                        save_directory: str="data/output/search_results.json"
                        ):
        ret = Retriever()
        ret.save_result(dataset_path, k, save_directory)

    def answer(self, query: str, k: int = 10):
        retriev = Retriever()
        chunks = retriev.piked_chunk(query, k)
        question_id = str(uuid.uuid4())
        respond = Answer(self.qwen, chunks, query, question_id)
        print(respond.generate_answer())

    def answer_dataset(self,
                       student_search_result_path: str,
                       save_directory: str="data/output/search_result_and_answer.json",
                       k: int=5
                       ):
        chunk = Retriever()

        with open(student_search_result_path, "r") as file:
            data = json.load(file)

        answers = data["search_result"]
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

        with open(save_directory, "w") as file:
            json.dump(output.model_dump(), file, indent=4)

    def evaluate(self,
                student_search_result_path: str,
                dataset_path: str,
                k: int = 10):
        pass
