from layers.layer4.config import Settings
from layers.layer4.llm.mock import MockBackend
from layers.layer4.review.l5_adapter import L5Reviewer
from layers.layer4.schemas.l5_context import L5Context
from layers.layer4.service import L4Service
from layers.layer5.reviewer import _l5_timeout_seconds, review_response
from shared.schemas.treatment_instructions import (
    BoundaryInstruction, DiagnosticAnchor, RoleInstruction,
    TreatmentInstructions, TurnInstruction,
)


def _context() -> L5Context:
    return L5Context(
        l4_raw_output="我们先一起看看最近的睡眠情况。",
        l4_substage="L4-L1",
        therapy_type="CBT",
        user_message="最近睡不好",
        history=[],
        instructions=TreatmentInstructions(
            L1=RoleInstruction(identity="CBT治疗师", style="温暖"),
            L2=DiagnosticAnchor(therapy_type="CBT"),
            L3=TurnInstruction(goal="探索睡眠困扰", technique="反映"),
            L4=BoundaryInstruction(absolute_bans=["禁止开药"]),
        ),
    )


def test_default_service_uses_real_local_l5_adapter():
    service = L4Service(
        Settings(_env_file=None, review_mode="l5_local"),
        backend=MockBackend(),
    )
    assert isinstance(service.reviewer, L5Reviewer)


def test_local_l5_adapter_fails_closed_when_l5_raises():
    def unavailable(_payload):
        raise TimeoutError("L5 unavailable")

    result = L5Reviewer(unavailable).review(_context(), [])

    assert result.passed is False
    assert result.harm == "fail"
    assert "TimeoutError" in result.fail_reason
    assert "不得放行" in result.fix_instruction


def test_mock_reviewer_requires_explicit_mode():
    from layers.layer4.review.mock_reviewer import MockReviewer

    service = L4Service(Settings(_env_file=None, review_mode="mock"))
    assert isinstance(service.reviewer, MockReviewer)


def test_l5_review_uses_default_60_second_timeout(monkeypatch):
    captured = {}

    def api(_prompt, **kwargs):
        captured.update(kwargs)
        return '{"harm":"pass","boundary":"pass","quality":"pass","fail_reason":"","fix_instruction":""}'

    monkeypatch.delenv("L5_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", api)

    result = review_response(_context().model_dump(mode="json"))

    assert result["harm"] == "pass"
    assert captured["timeout"] == 60.0


def test_l5_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("L5_LLM_TIMEOUT_SECONDS", "90")
    assert _l5_timeout_seconds() == 90.0


def test_l5_api_timeout_propagates_to_fail_closed_adapter(monkeypatch):
    def unavailable(_prompt, **_kwargs):
        raise TimeoutError("review API timeout")

    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", unavailable)
    result = L5Reviewer(review_response).review(_context(), [])

    assert result.passed is False
    assert "TimeoutError" in result.fail_reason
    assert "不得放行" in result.fix_instruction
