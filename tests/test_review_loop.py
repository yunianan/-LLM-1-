from l4.generation.responder import Responder
from l4.generation.summary_builder import build_l5_context
from l4.llm.base import Message
from l4.llm.mock import MockBackend
from l4.review.mock_reviewer import MockReviewer
from l4.review.orchestrator import DEFAULT_FALLBACK, ReviewOrchestrator
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.history import ConversationTurn

ROLE = RoleInstruction(identity="CBT治疗师", style="温暖")
ANCHOR = DiagnosticAnchor(primary="MDD", confidence=0.87, therapy_type="CBT")
BOUNDARY = BoundaryInstruction()
SAFETY = SafetyPassport()

TURN = TurnInstruction(goal="探索自动化思维", technique="苏格拉底式提问")

GOOD = '{"reply": "听起来这周很辛苦，是什么让你觉得自己必须这么做？", "stage_complete": true, "reason": "ok"}'
BAD = '{"reply": "你得了中度抑郁，建议你试试加大安眠药剂量。", "stage_complete": false, "reason": "bad"}'


def _make_orchestrator(mock_backend, mock_reviewer=None, max_attempts=3):
    responder = Responder(mock_backend, ROLE, ANCHOR, BOUNDARY)
    return ReviewOrchestrator(responder, mock_reviewer or MockReviewer(), max_attempts)


def _run(orchestrator, user_message="我这周又加班到凌晨，感觉撑不住了", validate=None):
    def build_ctx(reply):
        return build_l5_context(reply, "L4-L3", "CBT", SAFETY, ANCHOR)
    return orchestrator.run(TURN, [], user_message, build_ctx, validate)


def test_passes_first_try():
    orch = _make_orchestrator(MockBackend(default=GOOD))
    out = _run(orch)
    assert out.fallback_used is False
    assert out.attempts == 1
    assert len(out.review_history) == 1
    assert out.review_history[0].passed is True
    assert "觉得自己必须这么做" in out.reply


def test_fail_then_pass_uses_fix_instruction():
    backend = MockBackend(
        responses={
            "重写要求": '{"reply": "根据我们目前的评估，初步印象是情绪困扰，你感觉呢？", "stage_complete": false, "reason": "rewritten"}'
        },
        default=BAD,
    )
    orch = _make_orchestrator(backend, MockReviewer(fail_first=1))
    out = _run(orch)
    assert out.fallback_used is False
    assert out.attempts == 2
    assert len(out.fix_instructions) == 1
    assert "初步印象" in out.reply


def test_three_fails_fallback():
    backend = MockBackend(
        responses={"重写要求": BAD},
        default=BAD,
    )
    orch = _make_orchestrator(backend, MockReviewer(always_fail=True), max_attempts=3)
    out = _run(orch)
    assert out.fallback_used is True
    assert out.attempts == 3
    assert out.reply == DEFAULT_FALLBACK
    assert len(out.review_history) == 3
    assert all(not r.passed for r in out.review_history)


def test_validate_hook_fails_without_reviewer_fail():
    backend = MockBackend(
        responses={
            "请修正": '{"reply": "安全回复，不含禁止词。", "stage_complete": false, "reason": ""}'
        },
        default='{"reply": "这里出现了禁止词：建议", "stage_complete": false, "reason": ""}',
    )
    orch = _make_orchestrator(backend, MockReviewer(), max_attempts=2)

    def validate(reply: str) -> list[str]:
        return ["回复包含禁止词：建议"] if "建议" in reply else []

    out = _run(orch, validate=validate)
    assert out.fallback_used is False
    assert out.attempts == 2
    assert "安全回复" in out.reply
    assert "本地校验未通过" in out.review_history[0].fail_reason


def test_initial_fix_injected():
    backend = MockBackend(
        responses={"删除确定性措辞": '{"reply": "初步印象是情绪困扰，你感觉呢？", "stage_complete": false, "reason": "rewritten"}'},
        default=BAD,
    )
    orch = _make_orchestrator(backend, MockReviewer())

    def build_ctx(reply):
        return build_l5_context(reply, "L4-L3", "CBT", SAFETY, ANCHOR)

    # 不传 initial_fix：命中 default=BAD
    out_no_fix = orch.run(TURN, [], "我这周又加班到凌晨，感觉撑不住了", build_ctx)
    assert "你得了中度抑郁" in out_no_fix.reply

    # 传 initial_fix：首次生成即注入"删除确定性措辞"，命中 responses 键
    out = orch.run(TURN, [], "我这周又加班到凌晨，感觉撑不住了", build_ctx,
                   initial_fix="删除确定性措辞，改为初步印象")
    assert out.fallback_used is False
    assert out.attempts == 1
    assert "初步印象是情绪困扰" in out.reply
