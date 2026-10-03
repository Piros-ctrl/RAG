from pick_chunk import Indexer
from search import Retriever


path = "data/raw/vllm-0.10.1"
class CLI:
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
                        dataset: str,
                        k: int = 10,
                        save_directory: str="data/output/search_results"
                        ):
        pass

    def answer(self, query: str, k: int = 10):
        pass

    def answer_dataset(self,
                       student_search_result_path: str,
                       save_directory: str="data/output/search_result_and_answer"
                       ):
        pass

    def evaluate(self,
                student_search_result_path: str,
                dataset_path: str,
                k: int = 10):
        pass
