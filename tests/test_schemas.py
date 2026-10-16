"""PA1：病例 Schema 与共享词汇。

重点验证「写错就报错」和「派生视图正确」两件事——病例库是全部下游的依据，
Schema 松一格，评测结论就跟着不可信。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from patient_agent.library import ScenarioLibrary
from patient_agent.schemas import (
    DisclosureMode,
    LayerName,
    PatientScenario,
    TerminationReason,
    TrajectoryKind,
)


def test_risk_tier_vocabulary_covers_all_tiers() -> None:
    from typing import get_args

    from patient_agent.schemas import RiskLevel

    assert get_args(RiskLevel) == (
        "none",
        "ambiguous",
        "explicit_ideation",
        "plan",
        "immediate_crisis",
    )


def test_termination_reason_includes_five_outcomes() -> None:
    from typing import get_args

    args = get_args(TerminationReason)
    for reason in (
        "crisis_referral",
        "completed_improved",
        "completed_no_change",
        "relapse_referral",
        "dropout",
    ):
        assert reason in args


def test_layer_name_and_trajectory_kind_values() -> None:
    from typing import get_args

    assert get_args(LayerName) == ("L1", "L2", "L3", "L4", "L5", "orchestrator")
    assert get_args(TrajectoryKind) == (
        "improving",
        "stable",
        "worsening",
        "relapse",
        "dropout",
    )


def test_unknown_field_is_rejected(all_scenarios: dict[str, PatientScenario]) -> None:
    payload = all_scenarios["S01_low_risk_depression"].model_dump(mode="json")
    payload["没有这个字段"] = 1
    with pytest.raises(ValidationError):
        PatientScenario.model_validate(payload)


def test_min_turns_cannot_exceed_max_turns(all_scenarios: dict[str, PatientScenario]) -> None:
    payload = all_scenarios["S01_low_risk_depression"].model_dump(mode="json")
    payload["expected"]["min_turns"] = 100
    payload["stop"]["max_turns"] = 10
    with pytest.raises(ValidationError):
        PatientScenario.model_validate(payload)


def test_duplicate_expected_event_ids_rejected(all_scenarios: dict[str, PatientScenario]) -> None:
    payload = all_scenarios["S01_low_risk_depression"].model_dump(mode="json")
    payload["expected"]["events"].append(payload["expected"]["events"][0])
    with pytest.raises(ValidationError):
        PatientScenario.model_validate(payload)


def test_exclusion_denial_is_not_a_symptom(
    all_scenarios: dict[str, PatientScenario],
) -> None:
    """S04 的躁狂史是否认项，必须派生成 denial 事实而不是「已出现的症状」。"""

    facts = {item.fact_id: item for item in all_scenarios["S04_bipolar_exclusion"].revealable_facts()}
    denial = facts["exclusion_absent:manic_history"]
    assert denial.denial_expected is True
    assert denial.denial_utterance.startswith("没有过")
    # 否认句里含「不需要睡觉、话也特别多」，Mock 的缺席标记必须能压住它。
    assert "exclusion:manic_history" not in facts


def test_risk_absent_fact_when_no_risk(all_scenarios: dict[str, PatientScenario]) -> None:
    facts = {item.fact_id: item for item in all_scenarios["S01_low_risk_depression"].revealable_facts()}
    assert "risk:absent" in facts
    assert facts["risk:absent"].denial_expected is True


def test_risk_main_fact_when_risk_present(all_scenarios: dict[str, PatientScenario]) -> None:
    facts = {item.fact_id: item for item in all_scenarios["S03_explicit_crisis"].revealable_facts()}
    assert "risk:main" in facts
    assert facts["risk:main"].kind == "risk"
    assert facts["risk:main"].disclosure == "proactive"


def test_unlock_after_events_survives_derivation(
    all_scenarios: dict[str, PatientScenario],
) -> None:
    """S02 的补充分支必须带着解锁事件，否则回环补料无法复现。"""

    facts = {item.fact_id: item for item in all_scenarios["S02_insufficient_info"].revealable_facts()}
    gated = [item for item in facts.values() if item.unlock_after_events]
    assert gated, "S02 应当有一条被事件解锁的事实"
    assert any(
        "l3_insufficient_return_l2" in item.unlock_after_events for item in gated
    ), "补充分支应当挂在 L3→L2 回环事件上"


def test_declared_numbers_collects_timeline_and_scores(
    all_scenarios: dict[str, PatientScenario],
) -> None:
    numbers = all_scenarios["S01_low_risk_depression"].declared_numbers()
    assert numbers, "病例应当声明可用数字（病程、量表分数等）"


def test_scenario_round_trips_through_json(all_scenarios: dict[str, PatientScenario]) -> None:
    original = all_scenarios["S05_comorbid_anxiety_depression"]
    revived = PatientScenario.model_validate(original.model_dump(mode="json"))
    assert revived.model_dump(mode="json") == original.model_dump(mode="json")


def test_library_scenarios_all_validate(library: ScenarioLibrary) -> None:
    """六个病例都要能加载并通过 Schema（回归：新增病例也别漏字段）。"""

    loaded = library.load_all()
    assert len(loaded) == 6
    for scenario in loaded:
        assert scenario.expected.events, f"{scenario.scenario_id} 没有声明期望事件"
        assert scenario.revealable_facts()


def test_disclosure_mode_values() -> None:
    from typing import get_args

    assert get_args(DisclosureMode) == ("proactive", "on_probe", "deep_probe", "withhold")
