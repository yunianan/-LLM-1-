from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SupportedDisorder = Literal[
    "MDD",
    "GAD",
    "PTSD",
    "OCD",
    "Insomnia",
    "PanicDisorder",
    "SocialAnxietyDisorder",
    "AdjustmentDisorder",
    "SomaticSymptomDisorder",
    "Bipolar",
]

SUPPORTED_DISORDERS: tuple[str, ...] = (
    "MDD",
    "GAD",
    "PTSD",
    "OCD",
    "Insomnia",
    "PanicDisorder",
    "SocialAnxietyDisorder",
    "AdjustmentDisorder",
    "SomaticSymptomDisorder",
    "Bipolar",
)


class SymptomMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: int = Field(ge=0)
    total: int = Field(ge=1)

    @model_validator(mode="after")
    def matched_cannot_exceed_total(self) -> "SymptomMatch":
        if self.matched > self.total:
            raise ValueError("matched 不能大于 total")
        return self


class DiagnosticCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disorder: SupportedDisorder
    label: str = Field(min_length=1)
    symptom_match: SymptomMatch
    core_evidence_summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    considered_disorders: list[SupportedDisorder] = Field(min_length=10, max_length=10)
    candidates: list[DiagnosticCandidate] = Field(min_length=3, max_length=5)


class MissingDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    suggested_question: str = Field(min_length=1)


class CriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["met", "not_met", "pending"]
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class EliminativeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disorder: SupportedDisorder
    criteria_detail: list[CriterionAssessment] = Field(min_length=1)
    conclusion: Literal["meets_criteria", "watch", "ruled_out"]
    conclusion_reason: str = Field(min_length=1)
    missing_dimensions: list[MissingDimension] = Field(default_factory=list)


class VerifierDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "evidence_sufficiency",
        "logic_coherence",
        "differential_coverage",
        "no_reasoning_leaps",
    ]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class RewardModelAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    dimensions: list[VerifierDimension] = Field(min_length=4, max_length=4)
    fail_reasons: list[str] = Field(default_factory=list)
    missing_dimensions: list[MissingDimension] = Field(default_factory=list)
