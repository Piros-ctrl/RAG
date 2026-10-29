import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A chunk location given by file path and character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question with an id, but no answer or sources yet."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its expected sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of answered and unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for one question plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for all questions, retrieved with top k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with generated answers for all questions, with top k."""

    search_results: List[MinimalAnswer]
    k: int
