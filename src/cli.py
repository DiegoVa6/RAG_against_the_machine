import json
import pickle
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from .chunker import Chunker
from .models import (MinimalSource, UnansweredQuestion, AnsweredQuestion, 
                     RagDataset, MinimalSearchResults, MinimalAnswer, 
                     StudentSearchResults, StudentSearchResultsAndAnswer)
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

    def _read_json(self, file_path: str) -> Any | None:
        """Parse a JSON file, or return None if it can't be read.

        Every error is reported with a message instead of a traceback.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: file not found: {file_path}")
        except IsADirectoryError:
            print(f"Error: {file_path} is a directory, not a file")
        except PermissionError:
            print(f"Error: no permission to read {file_path}")
        except UnicodeDecodeError:
            print(f"Error: {file_path} is not a text file")
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} is not valid JSON ({e})")
        return None

    def index(self, max_chunk_size: int = 2000, overlap: int | None = None):
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

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 5):
        # 1. Check the arguments before doing any work
        if not isinstance(k, int) or k < 1:
            print("Error: k must be an integer >= 1")
            return
        save_directory = str(save_directory)
        if not save_directory.strip():
            print("Error: save_directory is empty")
            return

        # 2. Read and validate the dataset
        data = self._read_json(str(dataset_path))
        if data is None:
            return
        try:
            dataset = RagDataset.model_validate(data)
        except ValidationError as e:
            print(f"Error: {dataset_path} is not a valid dataset:\n{e}")
            return
        print(f"Loaded {len(dataset.rag_questions)} questions")

        index = self._load_index()
        if index is None:
            return

        # 3. Search every question: each Chunk becomes a MinimalSource
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = [
                MinimalSource(file_path=chunk.file_path,
                              first_character_index=chunk.start_index,
                              last_character_index=chunk.end_index)
                for chunk, _ in index.search(question.question, k)
            ]
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            ))
        results = StudentSearchResults(search_results=search_results, k=k)

        # 4. Save it with the same file name as the dataset
        output_path = Path(save_directory) / Path(str(dataset_path)).name
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(results.model_dump_json(indent=2))
        except OSError as e:
            print(f"Error: can't write {output_path} ({e})")
            return
        print(f"Saved student_search_results to {output_path}")

    def answer(self, query: str, k: int):
        from .llm import LLM


    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        from .llm import LLM


    def evaluate(self, student_search_results_path: str, dataset_path: str):
        pass
