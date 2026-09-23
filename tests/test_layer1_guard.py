from copy import deepcopy

import pytest

from layers.layer1 import detector
from layers.layer1.guard import L1ExitGuard, L1GuardError
from layers.layer1.policy_executor import execute_safety_policy


def _passport(**overrides):
    passport = {
        "layer_id": "L1",
        "safety_flag": "pass",
        "risk_level": 1,
        "risk_types": [],
        "crisis_flag": False,
        "sensitive_topics": [],
        "structured_labels": {"scale_skip_list": []},
        "flags": [],
        "instruction_to_next": "",
    }
    passport.update(overrides)
    passport["execution"] = execute_safety_policy(passport)
    return passport


def test_block_decision_executes_crisis_route_and_prevents_l2():
    passport = _passport(
        safety_flag="block",
        risk_level=3,
        crisis_flag=True,
        flags=["crisis_intervention"],
    )

    result = L1ExitGuard().assert_valid(passport)

    assert result["status"] == "pass"
    assert result["can_call_l2"] is False
    assert passport["execution"]["terminal"] is True
    assert passport["execution"]["crisis_response_generated"] is True
    assert "do_not_call_l2" in passport["execution"]["applied_controls"]


def test_warn_decision_executes_trauma_and_scale_controls():
    passport = _passport(
        safety_flag="warn",
        risk_level=2,
        sensitive_topics=["trauma"],
        flags=["avoid_trauma_probing", "monitor_suicidal_ideation"],
        structured_labels={"scale_skip_list": ["PCL-5"]},
    )

    result = L1ExitGuard().assert_valid(passport)

    assert result["can_call_l2"] is True
    assert set(result["verified_controls"]) == {
        "avoid_trauma_probing",
        "monitor_suicidal_ideation",
        "skip_scale:PCL-5",
    }


def test_guard_rejects_block_decision_that_was_not_executed():
    passport = _passport(safety_flag="block", risk_level=3, crisis_flag=True)
    tampered = deepcopy(passport)
    tampered["execution"] = {
        "route": "l2",
        "terminal": False,
        "crisis_response_generated": False,
        "applied_controls": [],
    }

    result = L1ExitGuard().verify(tampered)

    assert result["status"] == "fail"
    assert result["can_call_l2"] is False
    assert any("终止常规流程" in item for item in result["violations"])
    with pytest.raises(L1GuardError):
        L1ExitGuard().assert_valid(tampered)


def test_guard_rejects_inconsistent_crisis_fields():
    passport = _passport(safety_flag="pass", risk_level=3, crisis_flag=False)

    result = L1ExitGuard().verify(passport)

    assert result["status"] == "fail"
    assert result["can_call_l2"] is False


def test_safety_detect_runs_executor_then_guard(monkeypatch):
    monkeypatch.setattr(detector, "keyword_detect", lambda text: {
        "rule_score": 2,
        "risk_types": ["trauma"],
        "sensitive_topics": ["trauma"],
        "evidence_refs": [],
    })
    monkeypatch.setattr(detector, "llm_classify", lambda text: {
        "llm_score": 1,
        "risk_types": [],
        "confidence": 0.9,
        "evidence_refs": [],
    })
    monkeypatch.setattr(detector, "jailbreak_detect", lambda text: {
        "jailbreak_level": 0,
        "jailbreak_flag": False,
        "evidence_refs": [],
    })

    result = detector.safety_detect("测试")

    assert result["execution"]["route"] == "l2"
    assert "skip_scale:PCL-5" in result["execution"]["applied_controls"]
    assert result["guard"]["status"] == "pass"
    assert result["guard"]["can_call_l2"] is True
