from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter


class chunking:
    def exctract_paths(self, source_path):
        directory = Path(source_path)
        md = []
        py = []
        txt = []

        for path in list(directory.rglob("*")):
            if path.is_file() and path.suffix in [".md", ".py", ".txt"]:
                file_content = path.read_text(
                    encoding="utf-8", errors="replace"
                    )
                if path.suffix == ".md":
                    md.append({
                        "path": str(path),
                        "content": file_content})
                if path.suffix == ".py":
                    py.append({
                        "path": str(path),
                        "content": file_content})
                if path.suffix == ".txt":
                    txt.append({
                        "path": str(path),
                        "content": file_content})
        return {"md": md,
                "py": py,
                "txt": txt}

    def chunks(self, path, chunk_size):
        files = self.exctract_paths(path)
        md_txt_files = files["md"] + files["txt"]
        py_files = files["py"]
        result_source = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            add_start_index=True
        )

        for file in md_txt_files:
            paragraphs = splitter.create_documents([file["content"]])

            for doc in paragraphs:
                source = {
                    "file_path": file["path"],
                    "first_char": doc.metadata["start_index"],
                    "last_char": (
                        doc.metadata["start_index"]
                        + len(doc.page_content)
                        - 1
                    ),
                    "page_content": doc.page_content
                }
                result_source.append(source)

        splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            add_start_index=True
        )

        for file in py_files:
            paragraphs = splitter.create_documents([file["content"]])

            for doc in paragraphs:
                source = {
                    "file_path": file["path"],
                    "first_char": doc.metadata["start_index"],
                    "last_char": (
                        doc.metadata["start_index"]
                        + len(doc.page_content)
                        - 1
                    ),
                    "page_content": doc.page_content
                }
                result_source.append(source)

        return result_source
