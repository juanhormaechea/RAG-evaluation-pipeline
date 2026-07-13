"""Pydantic structured-output models for LLM grading, rewriting and judging."""
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check"""

    binary_score: str = Field(description="Relevance Score: 'yes' if relevant, 'no' if not relevant")


class RewrittenQuestion(BaseModel):
    """A reformulated question optimized for semantic vector-search retrieval"""

    rewritten_question: str = Field(description="The improved, standalone question. No preamble, explanation, or quotation marks.")


class JudgeGradingScheme(BaseModel):
    """Retrieval output grading scheme"""
    correctness: float = Field(description="Fraction of the reference's atomic facts correctly conveyed by the answer.", ge=0.0, le=1.0)

    nugget_recall: float = Field(description="For enumeration answers: covered expected items / total expected items, penalizing spurious extras.", ge=0.0, le=1.0)

    faithful: float = Field(description="Fraction of the answer's claims entailed by 'retrieved_context' (NLI: entailed / neutral / contradicted)", ge=0.0, le=1.0)

    retrieval: float = Field(description="Did the retrieved set contain the`source_documents` needed to answer?", ge=0.0, le=1.0)

    attribution: float = Field(description="Each sentence cites the supporting document (ALCE-style citation precision/recall)", ge=0.0, le=1.0)
