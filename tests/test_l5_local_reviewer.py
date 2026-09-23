from layers.layer4.config import Settings
from layers.layer4.review.l5_adapter import L5Reviewer
from layers.layer4.schemas.l5_context import L5Context
from layers.layer4.service import L4Service


def _context() -> L5Context:
    return L5Context(
        l4_raw_output="我们先一起看看最近的睡眠情况。",
        l4_substage="L4-L1",
        therapy_type="CBT",
    )


def test_default_service_uses_real_local_l5_adapter():
    service = L4Service(Settings(_env_file=None))
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
