import json
import pickle
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tqdm import tqdm

from .chunker import Chunker
from .models import (AnsweredDataset, MinimalSource,RagDataset, 
                     MinimalSearchResults, MinimalAnswer, 
                     StudentSearchResults, StudentSearchResultsAndAnswer)
from .bm25 import BM25Index
from .recall5 import question_recall


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 8000
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase. "
    "Answer using only the information in the numbered sources of the "
    "context. Be concise and quote exact names (functions, classes, flags, "
    "endpoints) as they appear in the sources. If the context does not "
    "contain the answer, say that you could not find it in the sources."
)


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

    def answer(self, query: str, k: int = 5):
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
            print("No relevant sources found: can't answer this question.")
            return
        messages = self._create_prompt(
            query, [(chunk.file_path, chunk.text) for chunk, _ in resultados])

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
            print(f"  {chunk.file_path} [{chunk.start_index}:{chunk.end_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str):
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
        output_path = (Path(save_directory)
                       / Path(student_search_results_path).name)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
        except OSError as e:
            print(f"Error: can't write {output_path} ({e})")
            return
        print(f"Saved student_search_results_and_answer to {output_path}")


    def evaluate(self, student_search_results_path: str, dataset_path: str):
        search_data = self._read_json(str(student_search_results_path))
        if search_data is None:
            return
        try:
            search_results = StudentSearchResults.model_validate(search_data)
        except ValidationError as e:
            print(f"Error: {student_search_results_path} "
                  f"is not a valid search results file:\n{e}")
            return
        print(f"Loaded {len(search_results.search_results)} search_result questions")

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
