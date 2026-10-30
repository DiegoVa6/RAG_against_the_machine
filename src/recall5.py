from .models import MinimalSource


def overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Return how many characters two ranges have in common."""
    s = max(start1, start2)
    e = min(end1, end2)
    return max(0, (e - s))


def iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Return the intersection over union of two ranges (0 to 1)."""
    inter = overlap(start1, end1, start2, end2)
    largo1 = end1 - start1
    largo2 = end2 - start2
    union = largo1 + largo2 - inter
    if union == 0:
        return 0.0
    return inter / union


def source_found(gold: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    """Tell if a retrieved source is in gold's file and overlaps it enough.

    Enough means an IoU above 0.05, the same threshold as the moulinette.
    """
    for source in retrieved:
        if gold.file_path != source.file_path:
            continue
        valor = iou(gold.first_character_index,
                    gold.last_character_index,
                    source.first_character_index,
                    source.last_character_index)
        if valor > 0.05:
            return True
    return False


def question_recall(gold_sources: list[MinimalSource],
                    retrieved: list[MinimalSource]) -> float:
    """Return the share of gold_sources found in retrieved (0 to 1)."""
    found = 0
    for gold_source in gold_sources:
        if source_found(gold_source, retrieved):
            found += 1
    try:
        result = (found/len(gold_sources))
    except ZeroDivisionError:
        result = 0.0
    return result
