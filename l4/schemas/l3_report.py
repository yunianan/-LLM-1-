from pydantic import BaseModel

from l4.schemas.instructions import DiagnosticAnchor


class PrimaryDiagnosis(BaseModel):
    disorder: str
    confidence: float = 0.0
    severity: str = ""
    dsm5_criteria_met: list[str] = []


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


class DiagnosticReport(BaseModel):
    """接口三：L3 → L4 DiagnosticReport（诊断报告）。未知字段忽略，真实 L3 完整 JSON 可解析。"""

    layer_id: str = "L3"
    timestamp: str = ""
    diagnosis: Diagnosis = Diagnosis()
    verifier_score: float = 0.0
    verifier_routing: str = ""
    recommended_therapy: RecommendedTherapy = RecommendedTherapy()
    severity_level: int = 1
    clinical_notes_for_L4: str = ""
    summary: str = ""
    structured_labels: StructuredLabels = StructuredLabels()
    evidence_refs: list[EvidenceRefIn] = []
    confidence: float = 0.0
    flags: list[str] = []
    instruction_to_next: str = ""

    def to_anchor(self) -> DiagnosticAnchor:
        primary = self.diagnosis.primary
        return DiagnosticAnchor(
            primary=f"{primary.disorder}（置信度 {primary.confidence}）" if primary else "",
            comorbidities=[s.disorder for s in self.diagnosis.secondary],
            excluded=[r.disorder for r in self.diagnosis.ruled_out],
            core_beliefs=[],
            plugins=list(self.recommended_therapy.addon),
            confidence=primary.confidence if primary else 0.0,
            therapy_type=self.recommended_therapy.primary,
        )
