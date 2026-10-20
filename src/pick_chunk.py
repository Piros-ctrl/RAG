import pickle, re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from chunking import chunking



_WORD = re.compile(r"\w+")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "how", "does", "do", "which", "that",
    "this", "with", "as", "by", "it", "be", "can",
}

def tokenize(text: str) -> List[str]:
    tokens = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())            # full form: get_top_n
        parts = [p.lower() for p in _PARTS.findall(word)]
        if len(parts) > 1:
            tokens.extend(parts)               # parts: get, top, n
    return [t for t in tokens if t not in STOPWORDS]


class Indexer:
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)

    def build(self, source_path: str, max_chunk_size: int) -> None:
        """Chunk the corpus, build a BM25 index, and persist it."""
        chunker = chunking()
        raw_chunks = chunker.chunks(
            source_path, max_chunk_size)

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
