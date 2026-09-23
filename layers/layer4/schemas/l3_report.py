from typing import Literal

from pydantic import BaseModel, Field

from layers.layer4.schemas.instructions import DiagnosticAnchor


class DSMCriterionEvidence(BaseModel):
    code: str
    name: str
    status: Literal["met", "not_met", "unknown"]
    met: bool
    evidence_refs: list[str] = Field(default_factory=list)


class PrimaryDiagnosis(BaseModel):
    disorder: str
    confidence: float = 0.0
    severity: str = ""
    evidence: str = ""
    dsm5_criteria_met: list[str] = []
    dsm5_criteria_detail: list[DSMCriterionEvidence] = Field(default_factory=list)


class SecondaryDiagnosis(BaseModel):
    disorder: str
    confidence: float = 0.0
    evidence: str = ""


class RuledOutDiagnosis(BaseModel):
    disorder: str
    reason: str = ""
    evidence_ref: str = ""


class Diagnosis(BaseModel):
    primary: PrimaryDiagnosis | None = None
    secondary: list[SecondaryDiagnosis] = []
    ruled_out: list[RuledOutDiagnosis] = []


class RecommendedTherapy(BaseModel):
    primary: str = ""
    addon: list[str] = []
    contraindications: list[str] = []


class TherapyRouting(BaseModel):
    primary: str = ""
    addon: list[str] = []
    l4_substage_priority: list[str] = []


class StructuredLabels(BaseModel):
    therapy_routing: TherapyRouting = TherapyRouting()
    wording_constraint: str = ""
    risk_monitoring: list[str] = []


class EvidenceRefIn(BaseModel):
    type: str = ""
    disorder: str = ""
    criteria_met: str = ""
    source_doc: str = ""
    threshold: str = ""
    therapy: str = ""
    source: str = ""
    criterion_code: str = ""
    criterion_name: str = ""
    status: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class MissingDimension(BaseModel):
    field: str
    purpose: str = ""
    suggested_question: str = ""


class VerifierCheck(BaseModel):
    check_id: str
    status: Literal["pass", "fail"]
    score: float
    weight: float
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class L3ValidationCheck(BaseModel):
    rule_id: str
    status: Literal["pass", "fail"]
    detail: str = ""


class L3ValidationReceipt(BaseModel):
    passed: bool
    route: str
    checks: list[L3ValidationCheck] = Field(default_factory=list)
    payload_hash: str
    policy_version: str


class DiagnosticReport(BaseModel):
    """接口三：L3 → L4 DiagnosticReport（诊断报告）。未知字段忽略，真实 L3 完整 JSON 可解析。"""

    layer_id: str = "L3"
    timestamp: str = ""
    diagnosis: Diagnosis = Diagnosis()
    provisional_diagnosis: PrimaryDiagnosis | None = None
    verifier_score: float = 0.0
    candidate_confidence: float = 0.0
    verifier_routing: str = ""
    verifier_checks: list[VerifierCheck] = Field(default_factory=list)
    verifier_fail_reasons: list[str] = Field(default_factory=list)
    recommended_therapy: RecommendedTherapy = RecommendedTherapy()
    severity_level: int = 1
    clinical_notes_for_L4: str = ""
    summary: str = ""
    structured_labels: StructuredLabels = StructuredLabels()
    evidence_refs: list[EvidenceRefIn] = []
    confidence: float = 0.0
    flags: list[str] = []
    instruction_to_next: str = ""
    missing_dimensions: list[MissingDimension] = []
    l3_validation: L3ValidationReceipt | None = None

    def to_anchor(self) -> DiagnosticAnchor:
        primary = self.diagnosis.primary
        return DiagnosticAnchor(
            primary=primary.disorder if primary else "",
            comorbidities=[s.disorder for s in self.diagnosis.secondary],
            excluded=[r.disorder for r in self.diagnosis.ruled_out],
            core_beliefs=[],
            plugins=list(self.recommended_therapy.addon),
            confidence=primary.confidence if primary else 0.0,
            candidate_confidence=(
                self.candidate_confidence or (primary.confidence if primary else 0.0)
            ),
            verifier_score=self.verifier_score,
            therapy_type=self.recommended_therapy.primary,
        )
