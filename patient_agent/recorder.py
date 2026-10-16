"""逐轮记录器：把「系统说了什么 / 患者说了什么 / 这一轮合不合法」存档。

记录器是评测（PA7）的唯一数据来源。它刻意保持「哑」——只追加，不判断对错，
判断在评测器里做。这样重跑评测不用重跑会话。
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.client.base import TurnResponse
from patient_agent.guard import Violation, GuardResult
from patient_agent.generator import PatientAction
from patient_agent.evidence import FactEvidence, AnswerSegment


class GenerationAttempt(BaseModel):
    """一次候选生成及复核的完整快照；不存密钥或内部思维链。"""
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    previous_attempt_id: str | None = None
    attempt_number: int
    doctor_question: str = ""
    selection_reason: str = ""
    selected_fact_id: str = ""
    authorized_evidence: list[FactEvidence] = Field(default_factory=list)
    generation_evidence: list[FactEvidence] = Field(default_factory=list)
    observed_events: list[str] = Field(default_factory=list)
    candidate_text: str = ""
    segments: list[AnswerSegment] = Field(default_factory=list)
    raw_output: str = ""
    generation_error: str = ""
    generated_at: str = ""
    generation_elapsed_ms: float = 0.0
    review_elapsed_ms: float = 0.0
    reviewed_at: str = ""
    model_id: str = ""
    prompt_version: str = ""
    prompt_system: str = ""
    prompt_user: str = ""
    action_payload: dict = Field(default_factory=dict)
    guard_result: GuardResult
    submission_status: str = "not_submitted"
    submitted_at: str = ""


class TurnRecord(BaseModel):
    """会话中的一轮完整快照。"""

    model_config = ConfigDict(extra="ignore")

    turn_index: int
    action_kind: str
    # 患者实际提交给系统的内容（说话文本，或量表作答）。
    patient_text: str = ""
    submitted: bool = True
    rejected_text: str = ""
    final_text: str = ""
    submission_status: str = "submitted"
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    response_payload: dict = Field(default_factory=dict)
    patient_fact_id: str = ""
    patient_denial: bool = False
    # 系统回复。
    system_reply: str = ""
    waiting_type: str = "none"
    main_layer: str = "L1"
    # 编排器可以把辅助层放在这里（例如 L3 出结论时把 L3 记为 L4 的 secondary）。
    secondary_layers: list[str] = Field(default_factory=list)
    route_intent: str = ""
    events: list[str] = Field(default_factory=list)
    # 本轮守卫结论。
    guard_ok: bool = True
    violations: list[dict] = Field(default_factory=list)
    # 重说了几次才通过。
    retries: int = 0
    error_code: str | None = None
    session_ended: bool = False
    termination_reason: str | None = None
    # 这一轮从「生成患者动作」到「拿到系统回复」的墙钟耗时（毫秒）。
    elapsed_ms: float = 0.0
    note: str = ""


class SessionRecorder(BaseModel):
    """一次运行的完整记录：用例元信息 + 逐轮明细 + 最终状态。"""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    scenario_version: str = ""
    scenario_title: str = ""
    session_id: str = ""
    seed: int = 0
    llm_mode: str = "offline"
    llm_name: str = ""
    system_api_name: str = ""
    turns: list[TurnRecord] = Field(default_factory=list)
    # 进入 L4 之后的轮数（量表逐题作答不影响这个计数，轨迹步用它对齐）。
    treatment_turns: int = 0
    # 会话结束时从系统读到的开发状态与总结。
    final_state: dict = Field(default_factory=dict)
    episode_summary: dict | None = None
    history: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # 运行器给出的停止原因（属于 patient agent 侧的测试停止，不是疗程终止）。
    stop_reason: str = ""

    # ------------------------------------------------------------------ 写入
    def add_turn(
        self,
        *,
        turn_index: int,
        action: PatientAction,
        patient_text: str,
        response: TurnResponse,
        guard_ok: bool = True,
        violations: list[Violation] | None = None,
        retries: int = 0,
        elapsed_ms: float = 0.0,
        note: str = "",
        attempts: list[GenerationAttempt] | None = None,
    ) -> TurnRecord:
        failed_request = response.route.intent == "error"
        record = TurnRecord(
            turn_index=turn_index,
            action_kind=action.kind,
            patient_text="" if failed_request else patient_text,
            submitted=not failed_request,
            final_text=patient_text,
            submission_status="request_failed_unknown" if failed_request else "submitted",
            attempts=[a.model_copy(deep=True) for a in (attempts or [])],
            response_payload=response.model_dump(mode="json"),
            patient_fact_id=action.fact_id,
            patient_denial=action.is_denial,
            system_reply=response.patient_reply,
            waiting_type=response.waiting_type,
            main_layer=response.route.main_layer,
            secondary_layers=list(response.route.secondary_layers),
            route_intent=response.route.intent,
            events=list(response.events),
            guard_ok=guard_ok,
            violations=[item.model_dump() for item in (violations or [])],
            retries=retries,
            error_code=response.error_code.value if response.error_code else None,
            session_ended=response.session_ended,
            termination_reason=response.termination_reason,
            elapsed_ms=elapsed_ms,
            note=note,
        )
        self.turns.append(record)
        return record

    def note(self, message: str) -> None:
        self.notes.append(message)

    # ------------------------------------------------------------------ 汇总
    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def all_events(self) -> list[str]:
        seen: list[str] = []
        for turn in self.turns:
            for event in turn.events:
                if event not in seen:
                    seen.append(event)
        return seen

    def layers_visited(self) -> list[str]:
        """走过哪些层，按 L1→L5 排序。

        辅助层也算「走过」：编排器常在给出 L4 结论的那一轮把 L3 放进
        ``secondary_layers``，只统计 main_layer 会漏掉 L3 的覆盖证据。
        """

        seen: set[str] = set()
        for turn in self.turns:
            if turn.main_layer:
                seen.add(turn.main_layer)
            seen.update(turn.secondary_layers)
        return sorted(seen)

    def guard_violations(self) -> list[dict]:
        items: list[dict] = []
        for turn in self.turns:
            for violation in turn.violations:
                items.append({"turn_index": turn.turn_index, **violation})
        return items

    def error_codes(self) -> list[str]:
        """出现过的系统错误码（按首次出现顺序，去重）。"""

        seen: list[str] = []
        for turn in self.turns:
            if turn.error_code and turn.error_code not in seen:
                seen.append(turn.error_code)
        return seen

    def termination_observed(self) -> str | None:
        """会话最终以什么原因结束；没有终止就是 None。"""

        for turn in self.turns:
            if turn.termination_reason:
                return turn.termination_reason
        return None

    def timing_summary(self) -> dict[str, float]:
        """逐轮耗时的汇总，供报告统计「模型耗时」。"""

        values = [turn.elapsed_ms for turn in self.turns if turn.elapsed_ms > 0]
        if not values:
            return {"count": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0}
        ordered = sorted(values)
        return {
            "count": len(values),
            "total_ms": round(sum(values), 3),
            "avg_ms": round(sum(values) / len(values), 3),
            "max_ms": round(ordered[-1], 3),
        }

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def dump(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "SessionRecorder":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
