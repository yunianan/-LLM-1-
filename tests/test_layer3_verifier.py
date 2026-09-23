import pytest

from layers.layer3.llm import CLINICRModelUnavailable
from layers.layer3.diagnosis.schemas import DiagnosticCandidate, EliminativeAssessment
from layers.layer3.diagnosis.verifier import IndependentVerifier
from tests.l3_test_double import ControlledRewardModelVerifier


def _candidate(confidence: float) -> DiagnosticCandidate:
    return DiagnosticCandidate.model_validate({
        "disorder": "MDD",
        "label": "重性抑郁障碍",
        "symptom_match": {"matched": 5, "total": 9},
        "core_evidence_summary": "五项标准有逐条证据",
        "evidence_refs": ["PHQ9_item2=3"],
        "confidence": confidence,
    })


def _assessment() -> EliminativeAssessment:
    return EliminativeAssessment.model_validate({
        "disorder": "MDD",
        "criteria_detail": [{
            "code": "A1",
            "name": "心境抑郁",
            "status": "met",
            "rationale": "单题和叙述均支持",
            "evidence_refs": ["PHQ9_item2=3"],
        }],
        "conclusion": "meets_criteria",
        "conclusion_reason": "进入终验",
        "missing_dimensions": [],
    })


def _profile() -> dict:
    return {"PHQ9": 16, "PHQ9_items": {"item2": 3}}


def test_verifier_score_is_independent_of_candidate_confidence():
    model = ControlledRewardModelVerifier(score=0.87)
    verifier = IndependentVerifier(model=model)

    low = verifier.verify(_candidate(0.1), _assessment(), _profile(), [])
    high = verifier.verify(_candidate(0.99), _assessment(), _profile(), [])

    assert low["score"] == high["score"] == 0.87
    assert low["routing"] == high["routing"] == "pass"
    assert all("confidence" not in call["candidate"] for call in model.calls)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, "pass"),
        (0.8, "pass"),
        (0.7999, "conditional"),
        (0.5, "conditional"),
        (0.4999, "return_to_l2"),
        (0.0, "return_to_l2"),
    ],
)
def test_verifier_uses_required_score_thresholds(score, expected):
    verifier = IndependentVerifier(model=ControlledRewardModelVerifier(score=score))

    result = verifier.verify(_candidate(0.75), _assessment(), _profile(), [])

    assert result["routing"] == expected
    assert {item["check_id"] for item in result["checks"]} == {
        "evidence_sufficiency",
        "logic_coherence",
        "differential_coverage",
        "no_reasoning_leaps",
    }


def test_conditional_verdict_requires_specific_l2_follow_up():
    class MissingFollowUpModel(ControlledRewardModelVerifier):
        def score_reasoning(self, *args, **kwargs):
            result = super().score_reasoning(*args, **kwargs)
            result.missing_dimensions = []
            return result

    verifier = IndependentVerifier(model=MissingFollowUpModel(score=0.6))

    with pytest.raises(CLINICRModelUnavailable, match="具体缺失信息"):
        verifier.verify(_candidate(0.75), _assessment(), _profile(), [])


def test_rejected_verdict_requires_reason_and_specific_l2_follow_up():
    class MissingReasonModel(ControlledRewardModelVerifier):
        def score_reasoning(self, *args, **kwargs):
            result = super().score_reasoning(*args, **kwargs)
            result.fail_reasons = []
            return result

    verifier = IndependentVerifier(model=MissingReasonModel(score=0.2))

    with pytest.raises(CLINICRModelUnavailable, match="拒绝原因"):
        verifier.verify(_candidate(0.75), _assessment(), _profile(), [])
