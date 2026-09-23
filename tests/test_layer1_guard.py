from copy import deepcopy

import pytest

from layers.layer1.guard import L1ExitGuard, L1GuardError
from layers.layer1.policy_executor import execute_safety_policy


def _passport(path="normal", *, instruction="允许进入L2。", flags=None):
    passport = {
        "decision_path": path,
        "instruction_to_next": instruction,
        "flags": flags or [],
    }
    passport["execution"] = execute_safety_policy(passport)
    return passport


def test_only_normal_path_can_call_l2():
    passport = _passport("normal")
    result = L1ExitGuard().assert_valid(passport)
    assert result["can_call_l2"] is True
    assert passport["execution"]["consume_message"] is False


@pytest.mark.parametrize(
    "path,route",
    [
        ("safety_assessment", "l1"),
        ("safety_pause", "safety"),
        ("clinical_crisis", "safety"),
        ("security_reject", "security"),
    ],
)
def test_non_normal_paths_consume_message_and_block_l2(path, route):
    passport = _passport(path, instruction="患者可见的下一步说明")
    result = L1ExitGuard().assert_valid(passport)
    assert passport["execution"]["route"] == route
    assert passport["execution"]["consume_message"] is True
    assert result["can_call_l2"] is False


def test_guard_rejects_assessment_message_forged_as_l2_input():
    passport = _passport("safety_assessment", instruction="请回答ASQ问题")
    tampered = deepcopy(passport)
    tampered["execution"]["route"] = "l2"
    tampered["execution"]["consume_message"] = False
    result = L1ExitGuard().verify(tampered)
    assert result["status"] == "fail"
    assert result["can_call_l2"] is False
    with pytest.raises(L1GuardError):
        L1ExitGuard().assert_valid(tampered)


def test_safety_pause_requires_patient_visible_transition_support():
    passport = _passport("safety_pause", instruction="")
    result = L1ExitGuard().verify(passport)
    assert result["status"] == "fail"
    assert any("支持" in item for item in result["violations"])


def test_guard_rejects_terminal_ui_shutdown():
    passport = _passport("clinical_crisis", instruction="支持与紧急行动")
    passport["execution"]["terminal"] = True
    assert L1ExitGuard().verify(passport)["status"] == "fail"
