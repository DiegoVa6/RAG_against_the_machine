import re
from dataclasses import dataclass

# A fenced code block opens/closes with ``` or ~~~ (up to 3 spaces before).
FENCE_RE = re.compile(r"^ {0,3}(```|~~~)", re.MULTILINE)


@dataclass
class Chunk:
    """A piece of a file, with its exact location in that file."""

    file_path: str
    text: str
    start_index: int
    end_index: int


def find_code_blocks(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) ranges of fenced code blocks in text.

    An unclosed block is considered to run until the end of the text.
    """
    blocks: list[tuple[int, int]] = []
    open_at: int | None = None
    for m in FENCE_RE.finditer(text):
        if open_at is None:
            open_at = m.start()            # opening fence
        else:
            blocks.append((open_at, m.end()))  # closing fence
            open_at = None
    if open_at is not None:
        blocks.append((open_at, len(text)))
    return blocks


def find_headings(
    text: str,
    start: int,
    end: int,
    level: int,
    code_blocks: list[tuple[int, int]],
) -> list[int]:
    """Return where headings of exactly this level start in [start, end).

    Lines inside fenced code blocks are ignored: there, "# ..." is a
    comment (bash, Python...), not a Markdown heading.
    """
    # "^## " matches "## Title" but not "### Title": after the two #,
    # a space is required.
    pattern = "^" + "#" * level + "[ \t]"
    positions: list[int] = []
    for m in re.finditer(pattern, text, re.MULTILINE):
        pos = m.start()
        if not start <= pos < end:
            continue
        if any(b_start <= pos < b_end for b_start, b_end in code_blocks):
            continue
        positions.append(pos)
    return positions


class Chunker:
    """Split files into chunks that keep their exact character offsets.

    Markdown files are split by headings; other files by fixed size.
    """

    def __init__(self, chunk_size: int = 2000, overlap: int = 200) -> None:
        """Configure the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            overlap: Characters shared between consecutive fixed-size
                chunks (only used when no structure is available).

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

    # ---------- Fixed size ----------

    def fixed_ranges(self, start: int, end: int) -> list[tuple[int, int]]:
        """Split [start, end) into windows of chunk_size with overlap."""
        ranges: list[tuple[int, int]] = []
        chunk_start = start
        chunk_end = start  # end of the last range created
        while chunk_end < end:
            chunk_end = min(chunk_start + self.chunk_size, end)
            ranges.append((chunk_start, chunk_end))
            chunk_start += self.chunk_size - self.overlap
        return ranges

    # ---------- Markdown ----------

    def markdown_ranges(
        self,
        text: str,
        start: int,
        end: int,
        level: int,
        code_blocks: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Recursively split text[start:end] by Markdown headings.

        1. If the range fits in chunk_size, it is one chunk (base case).
        2. Otherwise, cut it at the headings of the first level (from
           `level` down to 6) that has headings inside the range, and
           split each part again with the next level (recursive case).
        3. If no heading level is left, use fixed-size windows.
        """
        if end <= start:
            return []
        if end - start <= self.chunk_size:
            return [(start, end)]

        for lvl in range(level, 7):
            # A heading exactly at `start` is this section's own title:
            # cutting there would only produce an empty piece.
            cuts = [p for p in find_headings(text, start, end, lvl,
                                             code_blocks) if p > start]
            if not cuts:
                continue  # no headings of this level here: try deeper
            bounds = [start] + cuts + [end]
            ranges: list[tuple[int, int]] = []
            for part_start, part_end in zip(bounds, bounds[1:]):
                ranges.extend(self.markdown_ranges(
                    text, part_start, part_end, lvl + 1, code_blocks))
            return ranges

        return self.fixed_ranges(start, end)

    # ---------- Building chunks ----------

    def make_chunks(
        self, text: str, file_path: str, ranges: list[tuple[int, int]]
    ) -> list[Chunk]:
        """Turn (start, end) ranges of text into Chunk objects."""
        return [
            Chunk(file_path=file_path, text=text[s:e],
                  start_index=s, end_index=e)
            for s, e in ranges
        ]

    def chunk_text(self, text: str, file_path: str) -> list[Chunk]:
        """Split a text into fixed-size chunks."""
        return self.make_chunks(text, file_path,
                                self.fixed_ranges(0, len(text)))

    def chunk_markdown(self, text: str, file_path: str) -> list[Chunk]:
        """Split a Markdown text by its headings."""
        code_blocks = find_code_blocks(text)
        ranges = self.markdown_ranges(text, 0, len(text), 1, code_blocks)
        return self.make_chunks(text, file_path, ranges)

    def chunk_file(self, file_path: str) -> list[Chunk]:
        """Read a file and split it with the strategy for its type."""
        text = self.read_file(file_path)
        if file_path.endswith(".md"):
            return self.chunk_markdown(text, file_path)
        return self.chunk_text(text, file_path)

    def chunk_corpus(self, file_paths: list[str]) -> list[Chunk]:
        """Split every file in file_paths and return all chunks together."""
        chunks: list[Chunk] = []
        for file_path in file_paths:
            chunks.extend(self.chunk_file(file_path))
        return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into word terms (letters, digits, _)."""
    return re.findall(r"\w+", text.lower())
