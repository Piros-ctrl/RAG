import bm25s
from pathlib import Path
from typing import List
from tqdm import tqdm
from chunking import chunking


class Indexer:
    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)

    def build(self, source_path: str, max_chunk_size: int) -> None:
        """Chunk the corpus, build a BM25 index, and persist it."""
        chunker = chunking()
        raw_chunks = chunker.chunks(source_path, max_chunk_size)

        chunks: List[dict] = [
            chunk for chunk in tqdm(raw_chunks, desc="chunking", unit=" files")
        ]

        corpus_tokens = bm25s.tokenize(
            [chunk["page_content"] for chunk in chunks]
        )

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.processed_dir / "bm25_index"), corpus=chunks)

        print(f"Ingestion complete! Indexed {len(chunks)} chunks "
              f"under {self.processed_dir}/")