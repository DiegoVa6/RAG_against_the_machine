import json
import pickle
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .chunker import Chunker
from .models import (AnsweredDataset, MinimalSource, RagDataset,
                     MinimalSearchResults, MinimalAnswer,
                     StudentSearchResults, StudentSearchResultsAndAnswer)
from .bm25 import BM25Index
from .recall5 import question_recall


MODEL_NAME = "Qwen/Qwen3-0.6B"
# The moulinette rejects any source longer than 2000 characters
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 100
MAX_CONTEXT_CHARS = 8000
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase. "
    "Answer using only the information in the numbered sources of the "
    "context. Be concise and quote exact names (functions, classes, flags, "
    "endpoints) as they appear in the sources. If the context does not "
    "contain the answer, say that you could not find it in the sources."
)


def get_docs(root: str = "data/raw") -> list[str]:
    """Return the sorted paths of all .md, .txt and .py files under root."""
    extensions = {".md", ".txt", ".py"}
    return sorted(
        str(path)
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix in extensions
    )


class RagCLI:
    """Command-line interface of the RAG system, exposed with Python Fire.

    Each public method is a command: uv run python -m src <command>.
    Methods starting with "_" are helpers and are not shown as commands.
    """

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

    def _check_query(self, query: Any) -> str | None:
        """Return the query as text, or None (with a message) if empty.

        Fire guesses types from the input: `search 123` gives an int.
        """
        text = str(query)
        if not text.strip():
            print("Error: the query is empty")
            return None
        return text

    def _check_k(self, k: Any) -> bool:
        """Tell if k is a valid number of results, with a message if not."""
        if not isinstance(k, int) or k < 1:
            print("Error: k must be an integer >= 1")
            return False
        return True

    def _save_json(self, data: BaseModel, save_directory: str,
                   file_name: str, what: str) -> None:
        """Write data as JSON to save_directory/file_name and report it.

        Args:
            data: The pydantic model to save.
            save_directory: Folder to write into (created if needed).
            file_name: Name of the output file.
            what: Name of the content, for the final message.
        """
        output_path = Path(save_directory) / file_name
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(data.model_dump_json(indent=2))
        except OSError as e:
            print(f"Error: can't write {output_path} ({e})")
            return
        print(f"Saved {what} to {output_path}")

    def _read_source(self, source: MinimalSource,
                     cache: dict[str, str]) -> str:
        """Return the text a source points to ("" if it can't be read).

        Files are kept in cache: several sources often come from the
        same file, so each one is read from disk only once.
        """
        if source.file_path not in cache:
            try:
                with open(source.file_path, "r", encoding="utf-8") as f:
                    cache[source.file_path] = f.read()
            except (OSError, UnicodeDecodeError):
                cache[source.file_path] = ""
        text = cache[source.file_path]
        return text[source.first_character_index:
                    source.last_character_index]

    def _create_prompt(
        self, query: str, sources: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """Build the chat messages: instructions, numbered sources, question.

        sources is a list of (file_path, text), best first. They are added
        until MAX_CONTEXT_CHARS is reached, so the prompt stays within the
        model's token budget.
        """
        context = ""
        for i, (file_path, text) in enumerate(sources, start=1):
            source = f"[{i}] {file_path}\n{text}\n\n"
            if context and len(context) + len(source) > MAX_CONTEXT_CHARS:
                break
            context += source
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}Question: {query}"},
        ]

    def index(self, max_chunk_size: int = 2000,
              overlap: int | None = None) -> None:
        """Chunk data/raw/ and save a BM25 index under data/processed/.

        Args:
            max_chunk_size: Maximum characters per chunk (100 to 2000).
            overlap: Characters shared by consecutive fixed-size chunks.
                Defaults to 10% of max_chunk_size.
        """
        # Below MIN_CHUNK_SIZE the corpus becomes millions of chunks and
        # the index may not fit in memory.
        if (not isinstance(max_chunk_size, int)
                or not MIN_CHUNK_SIZE <= max_chunk_size <= MAX_CHUNK_SIZE):
            print(f"Error: max_chunk_size must be an integer between "
                  f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}")
            return
        if overlap is None:
            overlap = max_chunk_size // 10
        if not isinstance(overlap, int) or not 0 <= overlap < max_chunk_size:
            print("Error: overlap must be an integer >= 0 and smaller than "
                  "max_chunk_size")
            return
        docs = get_docs()
        if not docs:
            print("Error: no .md, .txt or .py files found in data/raw/")
            return
        chunker = Chunker(chunk_size=max_chunk_size, overlap=overlap)
        chunks = chunker.chunk_corpus(docs)
        if not chunks:
            print("Error: no file in data/raw/ could be read")
            return
        index = BM25Index(chunks)
        try:
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            with open("data/processed/index.pkl", "wb") as f:
                pickle.dump(index, f)
        except OSError as e:
            print(f"Error: can't save the index in data/processed/ ({e})")
            return
        print(f"Indexed {len(chunks)} chunks under data/processed/")

    def search(self, query: str, k: int = 5) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        checked_query = self._check_query(query)
        if checked_query is None or not self._check_k(k):
            return
        index = self._load_index()
        if index is None:
            return
        resultados = index.search(checked_query, k)
        if not resultados:
            print("No results found.")
            return
        for chunk, _ in resultados:
            print(f"{chunk.file_path} [{chunk.start_index}:{chunk.end_index}]")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 5) -> None:
        """Search every question of a dataset and save the results.

        The output is a StudentSearchResults JSON file with the same name
        as the dataset, inside save_directory.

        Args:
            dataset_path: JSON file with the questions (a RagDataset).
            save_directory: Folder where the results are written.
            k: Number of sources to return per question.
        """
        # 1. Check the arguments before doing any work
        if not self._check_k(k):
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
        self._save_json(results, save_directory,
                        Path(str(dataset_path)).name,
                        "student_search_results")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query with Qwen, using the top-k sources.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve as context.
        """
        checked_query = self._check_query(query)
        if checked_query is None or not self._check_k(k):
            return
        index = self._load_index()
        if index is None:
            return
        resultados = index.search(checked_query, k)
        if not resultados:
            print("No relevant sources found: can't answer this question.")
            return
        messages = self._create_prompt(
            checked_query,
            [(chunk.file_path, chunk.text) for chunk, _ in resultados])

        from .llm import LLM
        try:
            llm = LLM(MODEL_NAME)
            response = llm.generate_response(messages)
        except (OSError, RuntimeError) as e:
            print(f"Error: can't run the model {MODEL_NAME} ({e})")
            return
        print(response)
        print("\nSources:")
        for chunk, _ in resultados:
            print(f"  {chunk.file_path} "
                  f"[{chunk.start_index}:{chunk.end_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate an answer for every question of a search results file.

        The output is a StudentSearchResultsAndAnswer JSON file with the
        same name as the input file, inside save_directory.

        Args:
            student_search_results_path: JSON file made by search_dataset.
            save_directory: Folder where the answers are written.
        """
        student_search_results_path = str(student_search_results_path)

        save_directory = str(save_directory)
        if not save_directory.strip():
            print("Error: save_directory is empty")
            return

        data = self._read_json(student_search_results_path)
        if data is None:
            return
        try:
            search_results = StudentSearchResults.model_validate(data)
        except ValidationError as e:
            print(f"Error: {student_search_results_path} "
                  f"is not a valid search results file:\n{e}")
            return
        print(f"Loaded {len(search_results.search_results)} questions")

        from .llm import LLM
        try:
            llm = LLM(MODEL_NAME)
        except (OSError, RuntimeError) as e:
            print(f"Error: can't load the model {MODEL_NAME} ({e})")
            return

        answers: list[MinimalAnswer] = []
        cache: dict[str, str] = {}
        for result in tqdm(search_results.search_results,
                           desc="Answering"):
            # The JSON only has where each source is: read its text.
            sources = [
                (source.file_path, self._read_source(source, cache))
                for source in result.retrieved_sources
            ]
            sources = [(path, text) for path, text in sources if text]
            if not sources:
                answer = "No relevant sources found: can't answer."
            else:
                messages = self._create_prompt(result.question, sources)
                try:
                    answer = llm.generate_response(messages)
                except RuntimeError as e:
                    answer = f"Error while generating the answer ({e})"
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer,
            ))
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=search_results.k)

        # Save it with the same file name as the search results
        self._save_json(output, save_directory,
                        Path(student_search_results_path).name,
                        "student_search_results_and_answer")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print the recall@k of search results against a dataset.

        A correct source counts as found when a result is in the same file
        and overlaps it with an IoU above 0.05, like the moulinette.

        Args:
            student_search_results_path: JSON file made by search_dataset.
            dataset_path: JSON file with the questions and their correct
                sources (an AnsweredQuestions dataset).
        """
        search_data = self._read_json(str(student_search_results_path))
        if search_data is None:
            return
        try:
            search_results = StudentSearchResults.model_validate(search_data)
        except ValidationError as e:
            print(f"Error: {student_search_results_path} "
                  f"is not a valid search results file:\n{e}")
            return
        print(f"Loaded {len(search_results.search_results)} "
              "search_result questions")

        data = self._read_json(str(dataset_path))
        if data is None:
            return
        try:
            dataset = AnsweredDataset.model_validate(data)
        except ValidationError as e:
            print(f"Error: {dataset_path} is not a valid dataset with "
                  f"answers:\n{e}")
            return
        print(f"Loaded {len(dataset.rag_questions)} dataset questions")

        # Pair each result with its question in the dataset by question_id
        gold_by_id = {q.question_id: q for q in dataset.rag_questions}
        ks = [k for k in (1, 3, 5, 10) if k <= search_results.k]
        totals = {k: 0.0 for k in ks}  # sum of the recall of each question
        evaluated = 0
        for result in tqdm(search_results.search_results, desc="Evaluating"):
            gold = gold_by_id.get(result.question_id)
            if gold is None:
                continue  # this question is not in the dataset
            evaluated += 1
            for k in ks:
                totals[k] += question_recall(gold.sources,
                                             result.retrieved_sources[:k])

        if evaluated == 0:
            print("Error: no question of the results is in the dataset")
            return
        print(f"Questions evaluated: {evaluated}")
        for k in ks:
            print(f"Recall@{k}: {totals[k] / evaluated:.3f}")
