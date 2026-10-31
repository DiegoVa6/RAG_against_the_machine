import ast
import re
from dataclasses import dataclass

from tqdm import tqdm

# A fenced code block opens/closes with ``` or ~~~ (up to 3 spaces before).
FENCE_RE = re.compile(r"^ {0,3}(```|~~~)", re.MULTILINE)

# A word: letters, digits and _ (so load_lora_adapter is one word).
WORD_RE = re.compile(r"\w+")
# The parts of a CamelCase word: HTTPServer -> HTTP, Server;
# FusedMoE -> Fused, Mo, E; layer2 -> layer, 2.
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# Very common words that say nothing about the topic: they are in almost
# every question and chunk, so they only add noise to BM25. "vllm" is
# here too: the whole corpus is about it.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
    "on", "with", "what", "how", "does", "do", "which", "can", "be", "by",
    "as", "at", "it", "this", "that", "from", "vllm",
}


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


def find_line_starts(text: str) -> list[int]:
    """Return the offset where each line of text starts.

    line_starts[n - 1] is where line n starts (AST lines count from 1).
    The last element is len(text), so line_starts[n] is where line n
    ends, even for the last line.
    """
    line_starts: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line)
    line_starts.append(offset)
    return line_starts


class Chunker:
    """Split files into chunks that keep their exact character offsets.

    Markdown files are split by headings, Python files by their AST
    nodes, and other files by fixed size.
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

    # ---------- Python files ----------

    def pyfile_ranges(
        self,
        start: int,
        end: int,
        nodes: list[ast.stmt],
        line_starts: list[int],
    ) -> list[tuple[int, int]]:
        """Recursively split [start, end) of a Python file by AST nodes.

        1. If the range fits in chunk_size, it is one chunk (base case).
        2. If there are no nodes to cut at, use fixed-size windows.
        3. Otherwise, cut it at the end of each node, so blank lines,
           comments and decorators before a node fall in its piece.
           Each piece is split again: a class with the nodes of its
           body (its methods), anything else with no nodes.
        """
        if end <= start:
            return []
        if end - start <= self.chunk_size:
            return [(start, end)]
        if not nodes:
            return self.fixed_ranges(start, end)

        cuts: list[int] = []
        # The last node is never a cut: whatever follows it stays with it.
        for node in nodes[:-1]:
            end_line = node.end_lineno or node.lineno
            cuts.append(line_starts[end_line])
        bounds = [start] + cuts + [end]
        pieces = list(zip(bounds, bounds[1:]))  # one piece per node

        ranges: list[tuple[int, int]] = []
        for i in range(len(nodes)):
            s, e = pieces[i]
            node = nodes[i]
            if isinstance(node, ast.ClassDef):
                ranges.extend(self.pyfile_ranges(s, e, node.body,
                                                 line_starts))
            else:
                ranges.extend(self.pyfile_ranges(s, e, [], line_starts))
        return ranges

    # ---------- Merging ----------

    def merge_ranges(
        self, ranges: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Join consecutive ranges while the result fits in chunk_size.

        Only contiguous ranges are joined (one ends where the next
        starts), so the overlapping windows of fixed_ranges stay as they
        are.
        """
        if not ranges:
            return []
        merged: list[tuple[int, int]] = []
        box_start, box_end = ranges[0]  # the chunk being filled
        for s, e in ranges[1:]:
            if s == box_end and e - box_start <= self.chunk_size:
                box_end = e  # fits: add it to the current chunk
            else:
                merged.append((box_start, box_end))
                box_start, box_end = s, e
        merged.append((box_start, box_end))  # the last chunk
        return merged

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

    def chunk_pyfile(self, text: str, file_path: str) -> list[Chunk]:
        """Split a Python text by its classes, functions and statements.

        If the text is not valid Python, it is split by fixed size.
        """
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return self.chunk_text(text, file_path)
        line_starts = find_line_starts(text)
        ranges = self.pyfile_ranges(0, len(text), tree.body, line_starts)
        ranges = self.merge_ranges(ranges)
        return self.make_chunks(text, file_path, ranges)

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
        if file_path.endswith(".py"):
            return self.chunk_pyfile(text, file_path)
        if file_path.endswith(".md"):
            return self.chunk_markdown(text, file_path)
        return self.chunk_text(text, file_path)

    def chunk_corpus(self, file_paths: list[str]) -> list[Chunk]:
        """Split every file in file_paths and return all chunks together.

        A file that can't be read (not UTF-8 text, no permission...) is
        skipped with a warning, so one bad file doesn't stop the others.
        """
        chunks: list[Chunk] = []
        for file_path in tqdm(file_paths, desc="Chunking", unit="file"):
            try:
                chunks.extend(self.chunk_file(file_path))
            except UnicodeDecodeError:
                tqdm.write(f"Warning: skipped {file_path} "
                           "(not a UTF-8 text file)")
            except OSError as e:
                tqdm.write(f"Warning: skipped {file_path} ({e.strerror})")
        return chunks


def tokenize(text: str) -> list[str]:
    """Split a text into lowercase terms, also splitting identifiers.

    Each word is kept whole, and if it is an identifier its parts are
    added too: load_lora_adapter -> load_lora_adapter, load, lora, adapter.
    So a chunk matches a question that quotes the identifier and also one
    that only uses its words. Stopwords are removed at the end.
    """
    terms: list[str] = []
    for word in WORD_RE.findall(text):
        terms.append(word.lower())
        # Split snake_case first, then CamelCase inside each piece
        parts = [part.lower()
                 for piece in word.split("_")
                 for part in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            terms.extend(parts)
    return [term for term in terms if term not in STOPWORDS]
