import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A piece of a file, with its exact location in that file."""

    file_path: str
    text: str
    start_index: int
    end_index: int


class Chunker:
    """Split files into fixed-size character chunks with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        """Configure the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            overlap: Characters shared between consecutive chunks.

        Raises:
            ValueError: If chunk_size is not positive, or overlap is
                negative or not smaller than chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be >= 0 and < chunk_size "
                f"({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def read_file(self, file_path: str) -> str:
        """Return the full content of a file as text."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(self, text: str, file_path: str) -> list[Chunk]:
        """Split a text into chunks that keep their character offsets.

        Each chunk satisfies text[start_index:end_index] == chunk.text.
        """
        chunks: list[Chunk] = []
        start = 0
        end = 0  # end of the last chunk created
        while end < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                file_path=file_path,
                text=text[start:end],
                start_index=start,
                end_index=end,
            ))
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Read a file and split it into chunks."""
        return self.chunk_text(self.read_file(file_path), file_path)

    def chunk_corpus(self, file_paths: list[str]) -> list[Chunk]:
        """Split every file in file_paths and return all chunks together."""
        chunks: list[Chunk] = []
        for file_path in file_paths:
            chunks.extend(self.chunk_file(file_path))
        return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into word terms (letters, digits, _)."""
    return re.findall(r"\w+", text.lower())
