import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from chunking import chunking


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> List[str]:
    """Lowercase word/identifier tokenizer, splits snake_case-friendly."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Indexer:
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)

    def build(self, source_path: str, max_chunk_size: int) -> None:
        """Chunk the corpus, build a BM25 index, and persist it."""
        chunker = chunking()
        raw_chunks = chunker.chunks(
            max_chunk_size, source_path)

        chunks: List[dict] = [
            chunk for chunk in tqdm(raw_chunks, desc="chunking", unit=" files")
        ]

        tokenized_corpus = [
            tokenize(chunk["page_content"]) for chunk in chunks
            ]
        bm25 = BM25Okapi(tokenized_corpus)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(self.processed_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)

        print("Ingestion complete! Indexed "
              f"{len(chunks)} chunks under {self.processed_dir}/")
