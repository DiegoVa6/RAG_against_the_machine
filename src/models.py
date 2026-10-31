import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A range of characters [first, last) in a file of the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question to answer, with its id."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its correct sources and reference answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, with or without their answers."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class AnsweredDataset(BaseModel):
    """A dataset where every question must have its sources and answer.

    Without UnansweredQuestion as an alternative, a question with a broken
    source is a validation error instead of silently losing its sources.
    """

    rag_questions: List[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question and the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset: the results of every question."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset: results and answer of every question."""

    search_results: List[MinimalAnswer]
    k: int
