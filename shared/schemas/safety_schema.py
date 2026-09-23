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
    execution: dict[str, Any]
    guard: dict[str, Any]


class SafetyModelUnavailable(TypedDict):
    """L1 的 LLM 连续失败时返回的终止错误，不能传递给 L2。"""

    layer_id: Literal["L1"]
    status: Literal["error"]
    error_code: Literal["L1_LLM_UNAVAILABLE"]
    message: str
    terminal: Literal[True]
    can_call_l2: Literal[False]
    details: dict[str, Any]
    evidence_refs: List[EvidenceRef]
