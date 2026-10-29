from pathlib import Path
import pickle

from .chunker import Chunker
from .bm25 import BM25Index
from .recall5 import Recall


MODEL_PATH = "models/Qwen3-0.6B"
messages = [
    {
        "role": "user",
        "content": "What color is the sky?",
    }
]

def get_docs(root: str = "data/raw") -> list[str]:
    """Return the sorted paths of all .md and .txt files under root."""
    extensions = {".md", ".txt", ".py"}
    return sorted(
        str(path)
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix in extensions
    )


class RagCLI:
    def _load_index(self) -> BM25Index | None:
        """Load the index saved by `index`, or None if it can't be loaded."""
        try:
            with open("data/processed/index.pkl", "rb") as f:
                index: BM25Index = pickle.load(f)
                return index
        except FileNotFoundError:
            print("Error: index not found. Run 'index' first.")
        except (pickle.UnpicklingError, EOFError):
            print("Error: index is corrupted. Run 'index' again.")
        return None

    def index(self, max_chunk_size: int = 2000, overlap: int | None = None):
        # Fire passes "abc" as a str, so check the type before comparing.
        if (not isinstance(max_chunk_size, int)
                or not 1 <= max_chunk_size <= 2000):
            print("Error: max_chunk_size must be an integer between 1 and "
                  "2000")
            return
        if overlap is None:
            overlap = max_chunk_size // 10
        if not isinstance(overlap, int) or not 0 <= overlap < max_chunk_size:
            print("Error: overlap must be an integer >= 0 and smaller than "
                  "max_chunk_size")
            return
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        docs = get_docs()
        chunker = Chunker(chunk_size=max_chunk_size, overlap=overlap)
        chunks = chunker.chunk_corpus(docs)
        index = BM25Index(chunks)
        with open("data/processed/index.pkl", "wb") as f:
            pickle.dump(index, f)
        print(f"Indexed {len(chunks)} chunks under data/processed/")

    def search(self, query: str, k: int = 5):
        query = str(query)
        if not query.strip():
            print("Error: the query is empty")
            return
        if not isinstance(k, int) or k < 1:
            print("Error: k must be an integer >= 1")
            return
        index = self._load_index()
        if index is None:
            return
        resultados = index.search(query, k)
        if not resultados:
            print("No results found.")
            return
        for chunk, _ in resultados:
            print(f"{chunk.file_path} [{chunk.start_index}:{chunk.end_index}]")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        pass

    def answer(self, query: str, k: int):
        pass

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        pass

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        pass
