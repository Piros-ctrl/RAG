# import json
import pickle
from pathlib import Path
# from typing import List

# from tqdm import tqdm
# from pick_chunk import Indexer


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

    def sisi():
        pass
