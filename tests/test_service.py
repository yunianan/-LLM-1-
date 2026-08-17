import json
from pathlib import Path

import pytest

from l4.config import Settings
from l4.llm.mock import MockBackend
from l4.review.mock_reviewer import MockReviewer
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.service import L4Service, SessionRequest, TurnRequest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

ROLE = RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化")
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来"],
                               safety_trigger="用户表达自杀意图→立即触发安全协议")
SAFETY = SafetyPassport(risk_level=1, sensitive_topics=[], flags=["monitor_suicidal_ideation"])


def _service(**backend_kwargs) -> L4Service:
    backend = MockBackend(**backend_kwargs)
    return L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())


def _report() -> DiagnosticReport:
    raw = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    return DiagnosticReport.model_validate(raw)


def _new_session(service: L4Service, session_id: str | None = None) -> str:
    return service.create_session(
        SessionRequest(session_id=session_id, role=ROLE, report=_report(),
                       boundary=BOUNDARY, safety=SAFETY)
    )


def test_create_session_auto_id_and_explicit_id():
    service = _service()
    sid = _new_session(service)
    assert len(sid) == 32
    sid2 = _new_session(service, "my-session")
    assert sid2 == "my-session"


def test_turn_end_to_end():
    service = _service()
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="我最近工作压力很大，晚上总是睡不好。",
        turn=TurnInstruction(goal="建立共情，确认当前状态", technique="开放式提问+反映",
                             force_substage="L4-L1"),
    ))
    assert out.reply
    assert out.l4_substage == "L4-L1"
    assert out.fallback_used is False
    assert out.l5_context.l4_raw_output == out.reply
    assert out.l5_context.l4_substage == "L4-L1"
    assert out.l5_context.diagnosis_context.suicidal_ideation == "passive_only"
    assert out.summary.layer_id == "L4"
    assert out.summary.evidence_refs  # 非空
    assert out.summary.flags == []


def test_turn_unknown_session_raises():
    service = _service()
    with pytest.raises(KeyError):
        service.handle_turn(TurnRequest(
            session_id="nope", user_message="hi",
            turn=TurnInstruction(goal="g", technique="t"),
        ))


def test_forbidden_word_triggers_rewrite():
    backend = MockBackend(
        responses={
            "请修正": '{"reply": "那我们先把目标放在睡个好觉上，你觉得呢？", "stage_complete": false, "reason": ""}'
        },
        default='{"reply": "你的情况属于确诊的疾病，建议你换个工作。", "stage_complete": false, "reason": ""}',
    )
    service = L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="最近压力很大",
        turn=TurnInstruction(goal="建立共情", technique="反映",
                             forbidden=["建议", "确诊"]),
    ))
    assert out.fallback_used is False
    assert "建议" not in out.reply and "确诊" not in out.reply
    assert "本地校验未通过" in out.review_history[0].fail_reason


def test_fallback_marks_flags():
    backend = MockBackend(default='{"reply": "你得了抑郁，建议开药。", "stage_complete": false, "reason": ""}')
    service = L4Service(Settings(_env_file=None), backend=backend,
                        reviewer=MockReviewer(always_fail=True))
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="我感觉很糟",
        turn=TurnInstruction(goal="g", technique="t"),
    ))
    assert out.fallback_used is True
    assert "loop_A_fallback" in out.summary.flags
    assert "manual_review_required" in out.summary.flags
    assert out.reply == Settings(_env_file=None).fallback_reply


def test_handle_turn_initial_fix():
    backend = MockBackend(
        responses={"删除确定性措辞": '{"reply": "初步印象是情绪困扰，你感觉呢？", "stage_complete": false, "reason": ""}'},
        default='{"reply": "你得了中度抑郁。", "stage_complete": false, "reason": ""}',
    )
    service = L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())
    sid = _new_session(service)
    out = service.handle_turn(
        TurnRequest(session_id=sid, user_message="hi",
                    turn=TurnInstruction(goal="g", technique="t")),
        initial_fix="删除确定性措辞，改为初步印象",
    )
    assert "初步印象" in out.reply
