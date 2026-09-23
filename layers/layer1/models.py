from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TriState = Literal["yes", "no", "unknown", "refused"]
L1Phase = Literal["monitoring", "asq", "bssa", "safety_hold"]
SafetyDisposition = Literal[
    "not_assessed",
    "emergency_evaluation",
    "further_evaluation",
    "non_urgent_follow_up",
    "no_further_safety_intervention",
]


class SafetySignalAnalysis(BaseModel):
    """LLM 只判定 L1 三条顶层路线，不评估即时危险。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    suicide_signal_detected: bool
    jailbreak_detected: bool
    safe_to_enter_l2: bool
    confidence: float = Field(ge=0, le=1)
    evidence_quotes: list[str] = Field(default_factory=list)
    jailbreak_evidence: list[str] = Field(default_factory=list)
    asq_transition_support: str = Field(default="", max_length=80)
    summary: str

    @model_validator(mode="after")
    def validate_route_consistency(self) -> "SafetySignalAnalysis":
        expected_safe = not (self.suicide_signal_detected or self.jailbreak_detected)
        if self.safe_to_enter_l2 is not expected_safe:
            raise ValueError(
                "safe_to_enter_l2必须与自杀倾向/越狱风险判定一致"
            )
        transition = self.asq_transition_support.strip()
        if self.suicide_signal_detected and not transition:
            raise ValueError("检测到自杀倾向时必须生成简短ASQ接纳语")
        if not self.suicide_signal_detected and transition:
            raise ValueError("未检测到自杀倾向时不得生成ASQ接纳语")
        return self


class ASQState(BaseModel):
    """NIMH ASQ 2025 成人筛查状态；题目与判定由规则控制。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "not_started", "in_progress", "negative", "non_acute_positive",
        "acute_positive", "refused",
    ] = "not_started"
    trigger: Literal["baseline", "signal"] = "baseline"
    current_question: int = Field(default=0, ge=0, le=5)
    awaiting_attempt_timing: bool = False
    answers: dict[int, bool | None] = Field(default_factory=dict)
    most_recent_attempt_timing: str = ""


class BSSAState(BaseModel):
    """成人门诊 BSSA 的结构化事实与安全计划状态。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_started", "in_progress", "completed"] = "not_started"
    turn_count: int = Field(default=0, ge=0)
    current_ideation: TriState = "unknown"
    ideation_frequency: str = ""
    last_occurrence: str = ""
    plan_present: TriState = "unknown"
    plan_description: str = ""
    access_to_means: TriState = "unknown"
    intent_present: TriState = "unknown"
    past_attempts: TriState = "unknown"
    most_recent_attempt_timing: str = ""
    preparatory_behavior: TriState = "unknown"
    hopelessness: TriState = "unknown"
    impulsivity: TriState = "unknown"
    psychosis: TriState = "unknown"
    substance_impairment: TriState = "unknown"
    trusted_support_available: TriState = "unknown"
    reasons_for_living: list[str] = Field(default_factory=list)
    can_maintain_safety: TriState = "unknown"
    trusted_contact: str = ""
    coping_strategy: str = ""
    means_restriction: str = ""
    professional_contact: str = ""
    refused_fields: list[str] = Field(default_factory=list)
    evidence_quotes: dict[str, str] = Field(default_factory=dict)


class BSSAExtraction(BaseModel):
    """BSSA LLM 单轮结构化提取与下一问。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    current_ideation: TriState | None = None
    ideation_frequency: str | None = None
    last_occurrence: str | None = None
    plan_present: TriState | None = None
    plan_description: str | None = None
    access_to_means: TriState | None = None
    intent_present: TriState | None = None
    past_attempts: TriState | None = None
    most_recent_attempt_timing: str | None = None
    preparatory_behavior: TriState | None = None
    hopelessness: TriState | None = None
    impulsivity: TriState | None = None
    psychosis: TriState | None = None
    substance_impairment: TriState | None = None
    trusted_support_available: TriState | None = None
    reasons_for_living: list[str] | None = None
    can_maintain_safety: TriState | None = None
    trusted_contact: str | None = None
    coping_strategy: str | None = None
    means_restriction: str | None = None
    professional_contact: str | None = None
    refused_fields: list[str] = Field(default_factory=list)
    evidence_quotes: dict[str, str] = Field(default_factory=dict)
    next_question_target: str
    next_question: str


class SupportResponse(BaseModel):
    """安全过渡阶段的个性化支持回应。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    supportive_message: str = Field(min_length=20, max_length=360)


class L1SessionState(BaseModel):
    """L1 内部会话状态，不跨层传递。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    phase: L1Phase = "monitoring"
    baseline_asq_status: Literal["not_due", "due", "negative", "refused", "positive"] = "not_due"
    asq: ASQState = Field(default_factory=ASQState)
    bssa: BSSAState = Field(default_factory=BSSAState)
    disposition: SafetyDisposition = "not_assessed"
    last_signal: SafetySignalAnalysis | None = None
    active_controls: list[str] = Field(default_factory=list)
    last_support_message: str = ""
