from typing import Any, List, Literal, TypedDict


class EvidenceRef(TypedDict):
    source: str
    finding: str
    basis: str


class RiskScoreBreakdown(TypedDict):
    rule: int
    llm: int
    scale: int
    jailbreak: int


class SafetyPassport(TypedDict):
    layer_id: Literal["L1"]
    timestamp: str
    safety_flag: Literal["pass", "warn", "block"]
    risk_level: int
    risk_types: List[str]
    crisis_flag: bool
    sensitive_topics: List[str]
    summary: str
    structured_labels: dict[str, Any]
    evidence_refs: List[EvidenceRef]
    confidence: float
    flags: List[str]
    instruction_to_next: str
