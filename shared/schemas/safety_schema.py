from typing import Any, List, Literal, TypedDict


class EvidenceRef(TypedDict):
    source: str
    finding: str
    basis: str


class RiskScoreBreakdown(TypedDict):
    rule: int
    llm: int
    scale: int


class SecurityEvent(TypedDict):
    detected: bool
    event_type: str
    severity: int
    matched_patterns: List[str]


class SafetyPassport(TypedDict):
    schema_version: Literal["1.0"]
    layer_id: Literal["L1"]
    session_id: str
    message_id: str
    timestamp: str
    safety_flag: Literal["pass", "warn", "block"]
    risk_level: int
    risk_types: List[str]
    crisis_flag: bool
    decision_path: Literal[
        "clinical_crisis", "safety_assessment", "safety_pause",
        "security_reject", "normal"
    ]
    security_event: SecurityEvent
    sensitive_topics: List[str]
    summary: str
    structured_labels: dict[str, Any]
    evidence_refs: List[EvidenceRef]
    confidence: float
    flags: List[str]
    instruction_to_next: str
    execution: dict[str, Any]
    guard: dict[str, Any]


class SafetyModelUnavailable(TypedDict):
    """L1 的 LLM 连续失败时返回的终止错误，不能传递给 L2。"""

    schema_version: Literal["1.0"]
    layer_id: Literal["L1"]
    session_id: str
    message_id: str
    timestamp: str
    status: Literal["error"]
    error_code: Literal["L1_API_ERROR", "L1_MODEL_OUTPUT_INVALID"]
    message: str
    terminal: Literal[True]
    can_call_l2: Literal[False]
    details: dict[str, Any]
    evidence_refs: List[EvidenceRef]
