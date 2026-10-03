from pathlib import Path

# from .llm import LLM
from .chunker import Chunker


MODEL_PATH = "models/Qwen3-0.6B"
messages = [
    {
        "role": "user",
        "content": "What color is the sky?",
    }
]


def get_docs(root: str = "data/raw") -> list[str]:
    """Return the sorted paths of all .md and .txt files under root."""
    extensions = {".md", ".txt"}
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
    chunker = Chunker(chunk_size=2000, overlap=200)
    chunks = chunker.chunk_corpus(docs)
    print(len(chunks))
    print(chunks[34].text)


if __name__ == "__main__":
    main()
