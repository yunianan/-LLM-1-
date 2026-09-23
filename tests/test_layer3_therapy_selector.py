from __future__ import annotations

import pytest

from layers.layer3.api import DiagnoseRequest, build_report
from layers.layer3.output_validator import L3OutputValidator
from layers.layer3.therapy.therapy_selector import (
    BASE_THERAPY_BY_DIAGNOSIS,
    TherapySelector,
)
from layers.layer4.generation.prompt_assembler import assemble_prompt
from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    TurnInstruction,
)
from layers.layer4.schemas.l3_report import DiagnosticReport


@pytest.mark.parametrize(
    ("diagnosis", "expected"),
    list(BASE_THERAPY_BY_DIAGNOSIS.items()),
)
def test_step1_maps_all_ten_diagnoses(diagnosis, expected):
    assert TherapySelector().select_basic(diagnosis) == expected


def test_step1_rejects_unknown_diagnosis_instead_of_silently_using_cbt():
    with pytest.raises(ValueError, match="不支持的主诊断"):
        TherapySelector().select_basic("Unknown")


@pytest.mark.parametrize(
    ("diagnosis", "expected"),
    list(BASE_THERAPY_BY_DIAGNOSIS.items()),
)
def test_step1_all_ten_mappings_pass_l3_output_guard(diagnosis, expected):
    therapy = TherapySelector().run(
        diagnosis,
        {"PHQ9": 5, "GAD7": 5},
        confirmed_diagnoses={diagnosis},
    )
    clinicr_result = {
        "diagnosis": {
            "primary": {
                "disorder": diagnosis,
                "confidence": 0.8,
                "severity": "mild",
                "evidence": "evidence",
                "dsm5_criteria_met": ["C1"],
                "dsm5_criteria_detail": [{
                    "code": "C1",
                    "name": "criterion",
                    "status": "met",
                    "met": True,
                    "evidence_refs": ["evidence-1"],
                }],
            },
            "secondary": [],
            "under_observation": [],
            "ruled_out": [],
        },
        "verifier_score": 0.9,
        "verifier_decision": "confirmed",
        "verifier_routing": "pass",
        "verifier_checks": [],
        "verifier_fail_reasons": [],
        "severity_level": 1,
        "evidence_refs": [],
        "summary": "therapy routing",
        "confidence": 0.9,
        "flags": [],
    }
    raw = build_report({"PHQ9": 5, "GAD7": 5}, clinicr_result, therapy)
    receipt = L3OutputValidator().validate(raw)

    assert raw["recommended_therapy"]["primary"] == expected
    assert receipt["passed"] is True


def test_step2_applies_priority_and_limits_active_plugins_to_two():
    therapy = TherapySelector().run(
        "Bipolar",
        {
            "MDQ": 8,
            "rhythm_disturbance": True,
            "PCL5": 40,
            "exposure_avoidant": True,
            "Y_BOCS": 20,
            "ritual_behaviors": True,
            "PHQ9": 12,
            "GAD7": 10,
        },
        confirmed_diagnoses={"Bipolar", "PTSD", "OCD"},
        verifier_decision="confirmed",
    )

    assert therapy["plugins"] == ["IPSRT", "EMDR"]
    assert [item["plugin"] for item in therapy["plugin_triggers"]] == [
        "IPSRT",
        "EMDR",
    ]
    assert all(item["conditions"] for item in therapy["plugin_triggers"])


def test_step2_uses_confirmed_comorbidity_for_diagnosis_specific_plugin():
    therapy = TherapySelector().run(
        "GAD",
        {"PCL5": 40, "exposure_avoidant": True},
        confirmed_diagnoses={"GAD", "PTSD"},
        verifier_decision="confirmed",
    )
    assert therapy["plugins"] == ["EMDR"]
    assert "不推荐EMDR（无PTSD诊断证据）" not in therapy["contraindications"]


def test_step2_does_not_activate_diagnosis_specific_plugin_for_provisional_case():
    therapy = TherapySelector().run(
        "PTSD",
        {"PCL5": 40, "exposure_avoidant": True},
        confirmed_diagnoses={"PTSD"},
        verifier_decision="provisional",
    )
    assert therapy["plugins"] == []


def test_step2_cbt_i_uses_insomnia_chief_complaint_not_primary_diagnosis():
    therapy = TherapySelector().run(
        "MDD",
        {
            "ISI": 18,
            "chronic_months": 4,
            "insomnia_chief_complaint": True,
        },
        confirmed_diagnoses={"MDD"},
    )
    assert therapy["plugins"] == ["CBT-I"]


def test_step2_sfbt_requires_confirmed_adjustment_disorder():
    profile = {"ADNM": 50, "single_focus": True}
    selector = TherapySelector()

    without_diagnosis = selector.run(
        "GAD",
        profile,
        confirmed_diagnoses={"GAD"},
    )
    with_comorbidity = selector.run(
        "GAD",
        profile,
        confirmed_diagnoses={"GAD", "AdjustmentDisorder"},
    )

    assert without_diagnosis["plugins"] == []
    assert with_comorbidity["plugins"] == ["SFBT"]


def test_step3_severe_and_high_motivation_adjustments_are_both_preserved():
    therapy = TherapySelector().run(
        "MDD",
        {
            "PHQ9": 20,
            "GAD7": 6,
            "motivation": {"importance": 8, "confidence": 6},
        },
        confirmed_diagnoses={"MDD"},
    )
    adjustments = therapy["strategy_adjustments"]

    assert adjustments["severity_band"] == "severe"
    assert adjustments["technical_density"] == "reduced"
    assert adjustments["supportive_response"] == "increased"
    assert adjustments["safety_monitoring"] == "enhanced"
    assert adjustments["motivation_band"] == "high"
    assert adjustments["challenge_interventions"] == "early_allowed"


def test_step3_low_motivation_defers_action_oriented_interventions():
    therapy = TherapySelector().run(
        "GAD",
        {
            "PHQ9": 8,
            "GAD7": 7,
            "motivation": {"importance": 3, "confidence": 4},
        },
        confirmed_diagnoses={"GAD"},
    )
    adjustments = therapy["strategy_adjustments"]

    assert adjustments["severity_band"] == "mild"
    assert adjustments["motivation_band"] == "low"
    assert adjustments["challenge_interventions"] == "deferred"
    assert "behavioral_activation" in adjustments["intervention_focus"]


def test_step3_missing_scales_remain_undetermined_instead_of_becoming_mild():
    therapy = TherapySelector().run(
        "MDD",
        {},
        confirmed_diagnoses={"MDD"},
    )

    assert DiagnoseRequest().PHQ9 is None
    assert DiagnoseRequest().GAD7 is None
    assert therapy["strategy_adjustments"]["severity_band"] == "undetermined"


def test_step3_severe_dissociation_suppresses_exposure_plugin():
    therapy = TherapySelector().run(
        "PTSD",
        {
            "PCL5": 40,
            "exposure_avoidant": True,
            "severe_dissociation": True,
        },
        confirmed_diagnoses={"PTSD"},
    )

    assert therapy["plugins"] == []
    assert therapy["strategy_adjustments"]["exposure_policy"] == "prohibited_until_stabilized"
    assert therapy["strategy_adjustments"]["suppressed_plugins"] == [{
        "plugin": "EMDR",
        "reason": "严重解离时先稳定化，不进行暴露",
    }]


def test_step3_active_substance_dependence_suppresses_all_plugins():
    therapy = TherapySelector().run(
        "GAD",
        {
            "AAQ_II": 30,
            "substance_active": True,
        },
        confirmed_diagnoses={"GAD"},
    )

    assert therapy["plugins"] == []
    assert therapy["strategy_adjustments"]["care_priority"] == "addiction_treatment_first"


def test_step3_psychotic_symptoms_prioritize_psychiatric_referral():
    therapy = TherapySelector().run(
        "MDD",
        {
            "motivation": {"importance": 8, "confidence": 8},
            "psychotic_symptoms": True,
        },
        confirmed_diagnoses={"MDD"},
    )
    adjustments = therapy["strategy_adjustments"]

    assert adjustments["care_priority"] == "psychiatric_referral"
    assert "psychiatric_referral" in adjustments["care_priorities"]
    assert adjustments["challenge_interventions"] == "deferred"
    assert "合并精神病性症状，建议精神科就医" in adjustments["instructions"]


def test_step3_contraindication_overrides_high_motivation_exposure_permission():
    therapy = TherapySelector().run(
        "PTSD",
        {
            "motivation": {"importance": 8, "confidence": 8},
            "severe_dissociation": True,
        },
        confirmed_diagnoses={"PTSD"},
    )
    adjustments = therapy["strategy_adjustments"]

    assert adjustments["challenge_interventions"] == "deferred"
    assert adjustments["exposure_policy"] == "prohibited_until_stabilized"
    assert "可较早引入暴露练习或行为实验等挑战性干预" not in adjustments["instructions"]


def test_therapy_output_survives_l3_to_l4_schema_and_prompt():
    therapy = TherapySelector().run(
        "MDD",
        {
            "PHQ9": 20,
            "GAD7": 6,
            "relapse_count": 3,
            "motivation": {"importance": 8, "confidence": 6},
        },
        confirmed_diagnoses={"MDD"},
    )
    clinicr_result = {
        "diagnosis": {
            "primary": {
                "disorder": "MDD",
                "confidence": 0.8,
                "severity": "severe",
                "evidence": "evidence",
                "dsm5_criteria_met": ["A1"],
                "dsm5_criteria_detail": [{
                    "code": "A1",
                    "name": "情绪低落",
                    "status": "met",
                    "met": True,
                    "evidence_refs": ["PHQ9_item2=3"],
                }],
            },
            "secondary": [],
            "under_observation": [],
            "ruled_out": [],
        },
        "verifier_score": 0.9,
        "verifier_decision": "confirmed",
        "verifier_routing": "pass",
        "verifier_checks": [],
        "verifier_fail_reasons": [],
        "severity_level": 3,
        "evidence_refs": [],
        "summary": "MDD治疗匹配",
        "confidence": 0.9,
        "flags": [],
    }
    raw = build_report(
        {"PHQ9": 20},
        clinicr_result,
        therapy,
    )
    raw["l3_validation"] = L3OutputValidator().validate(raw)
    report = DiagnosticReport.model_validate(raw)
    anchor = report.to_anchor()

    assert report.recommended_therapy.addon == ["MBCT"]
    assert report.recommended_therapy.plugin_triggers[0]["plugin"] == "MBCT"
    assert anchor.strategy_adjustments["severity_band"] == "severe"

    messages = assemble_prompt(
        role=RoleInstruction(identity="心理支持助手", style="温暖"),
        anchor=anchor,
        boundary=BoundaryInstruction(),
        turn=TurnInstruction(goal="支持患者", technique="CBT"),
        history=[],
        user_message="今天很疲惫",
    )
    assert "MBCT" in messages[0].content
    assert "降低每轮技术密度" in messages[0].content
    assert raw["l3_validation"]["passed"] is True
