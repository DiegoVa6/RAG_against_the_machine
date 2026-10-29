from pathlib import Path

# from .llm import LLM
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


def main():
    """
    Test LLMs
    """
#    llm = LLM(MODEL_PATH)
#    response = llm.generate_response(messages)
#    print(response)
    docs = get_docs()
    chunker = Chunker(chunk_size=1000, overlap=100)
    chunks = chunker.chunk_corpus(docs)
    print(len(chunks))
    bmindex = BM25Index(chunks)
    recall = Recall(bmindex)
    dataset = recall.read_file(
        "data/datasets/AnsweredQuestions/dataset_docs_public.json")
    recall.check_retrieval(dataset, 5)

#     resultados = bmindex.search("how to install vllm", k=5)
#     for chunk, nota in resultados:
#         print(nota, chunk.file_path, chunk.start_index, chunk.end_index)


if __name__ == "__main__":
    main()
