import math
from collections import Counter

from .chunker import Chunk, tokenize


class BM25Index:
    """Lexical index over chunks, ranked with the BM25 function."""

    def __init__(
        self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75
    ) -> None:
        """Build the index: term counts, lengths and IDF of every term.

        Args:
            chunks: Chunks to index. Their position in the list is their id.
            k1: Term frequency saturation (higher = repetitions count more).
            b: Length normalization (0 = none, 1 = full).
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Parallel lists: position i always refers to chunks[i].
        self.term_counts: list[Counter[str]] = []  # TF of each chunk
        self.lengths: list[int] = []               # terms in each chunk

        # DF: in how many chunks each term appears (at least once).
        doc_freq: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self.term_counts.append(Counter(tokens))
            self.lengths.append(len(tokens))
            doc_freq.update(set(tokens))  # set: count each term once

        self.n_chunks = len(chunks)
        total_length = sum(self.lengths)
        # "or 1.0" avoids dividing by zero if there are no terms at all.
        self.avg_length = (total_length / self.n_chunks
                           if total_length else 1.0)

        # IDF: rare terms weigh a lot, common terms almost nothing.
        self.idf: dict[str, float] = {
            term: math.log((self.n_chunks - df + 0.5) / (df + 0.5) + 1)
            for term, df in doc_freq.items()
        }

    def score(self, query_terms: list[str], i: int) -> float:
        """Return the BM25 score of chunk i for the given query terms."""
        counts = self.term_counts[i]
        # > 1 if the chunk is longer than average, < 1 if shorter.
        length_norm = 1 - self.b + self.b * self.lengths[i] / self.avg_length
 
        total = 0.0
        for term in query_terms:
            tf = counts[term]
            if tf == 0:
                continue  # term not in this chunk: adds nothing
            saturated_tf = (tf * (self.k1 + 1)
                            / (tf + self.k1 * length_norm))
            total += self.idf[term] * saturated_tf
        return total

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs, best first.

        Chunks with score 0 (no query term in common) are not returned.
        """
        query_terms = tokenize(query)
        if k <= 0 or not query_terms:
            return []

        scores = [self.score(query_terms, i) for i in range(self.n_chunks)]
        best = sorted(range(self.n_chunks),
                      key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in best if scores[i] > 0]
