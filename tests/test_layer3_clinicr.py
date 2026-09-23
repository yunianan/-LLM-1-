from layers.layer3.clinicr import CLINICR
from layers.layer3.api import diagnose
from layers.layer4.schemas.l3_report import DiagnosticReport


def test_phq9_total_score_does_not_prove_individual_dsm5_criteria():
    """量表总分只能用于筛查/严重度，不能证明任何 DSM-5 单项症状。"""
    profile = {
        "PHQ9": 20,
        # 故意不提供 PHQ-9 单题结果、患者原话、持续时间或功能损害证据。
    }

    result = CLINICR().run(profile)
    primary = result["diagnosis"]["primary"]

    assert primary is None
    assert result["verifier_routing"] == "return_to_l2"
    assert result["verifier_score"] == 0.0
    assert "insufficient_diagnostic_evidence" in result["flags"]
    missing_fields = {item["field"] for item in result["missing_dimensions"]}
    assert "phq9_items.item1" in missing_fields
    assert "phq9_items.item2" in missing_fields
    assert "symptom_duration_days" in missing_fields
    assert "functional_impairment" in missing_fields


def test_mdd_criteria_are_derived_from_items_and_patient_evidence():
    profile = {
        "PHQ9": 16,
        "PHQ9_items": {
            "item1": 3,
            "item2": 3,
            "item3": 2,
            "item4": 2,
            "item5": 0,
            "item6": 2,
            "item7": 0,
            "item8": 0,
            "item9": 0,
        },
        "symptom_duration_days": 21,
        "functional_impairment": True,
        "mania_or_hypomania_history": False,
        "substance_or_medical_cause": False,
        "differential_diagnoses_assessed": ["Bipolar", "AdjustmentDisorder"],
        "evidence_conflicts": [],
    }

    result = CLINICR().run(profile)
    primary = result["diagnosis"]["primary"]

    assert primary is not None
    assert primary["dsm5_criteria_met"] == ["A1", "A2", "A4", "A6", "A7"]
    assert result["verifier_routing"] == "pass"
    assert result["verifier_score"] == 1.0

    details = {item["code"]: item for item in primary["dsm5_criteria_detail"]}
    assert set(details) == {f"A{i}" for i in range(1, 10)}
    for code in primary["dsm5_criteria_met"]:
        assert details[code]["status"] == "met"
        assert details[code]["evidence_refs"]
    assert details["A1"]["evidence_refs"] == ["PHQ9_item2=3"]
    assert details["A2"]["evidence_refs"] == ["PHQ9_item1=3"]


def test_diagnose_returns_actionable_l2_feedback_when_evidence_is_missing():
    report = diagnose({"PHQ9": 20})

    assert report["diagnosis"]["primary"] is None
    assert report["verifier_routing"] == "return_to_l2"
    assert report["recommended_therapy"]["primary"] == ""
    assert report["instructions"]["L2"]["action"] == "collect_missing_dimensions"
    assert report["missing_dimensions"]
    assert all(item["suggested_question"] for item in report["missing_dimensions"])
    assert report["provisional_diagnosis"]["dsm5_criteria_detail"]


def test_dsm_criterion_evidence_survives_l3_to_l4_schema_parsing():
    report = diagnose({
        "PHQ9": 16,
        "PHQ9_items": {
            "item1": 3, "item2": 3, "item3": 2, "item4": 2, "item5": 0,
            "item6": 2, "item7": 0, "item8": 0, "item9": 0,
        },
        "key_symptoms_extracted": [{
            "symptom": "depressed_mood",
            "sources": ["patient_utterance_L008"],
        }],
        "symptom_duration_days": 21,
        "functional_impairment": True,
        "mania_or_hypomania_history": False,
        "substance_or_medical_cause": False,
        "differential_diagnoses_assessed": ["Bipolar", "AdjustmentDisorder"],
        "evidence_conflicts": [],
    })

    parsed = DiagnosticReport.model_validate(report)
    primary = parsed.diagnosis.primary
    assert primary is not None
    assert parsed.candidate_confidence == 0.75
    assert parsed.verifier_score == 1.0
    anchor = parsed.to_anchor()
    assert anchor.candidate_confidence == 0.75
    assert anchor.verifier_score == 1.0
    details = {item.code: item for item in primary.dsm5_criteria_detail}
    assert details["A1"].evidence_refs == ["PHQ9_item2=3", "patient_utterance_L008"]

    criterion_refs = {
        item.criterion_code: item
        for item in parsed.evidence_refs
        if item.type == "DSM-5-criterion"
    }
    assert criterion_refs["A1"].status == "met"
    assert criterion_refs["A1"].evidence_refs == [
        "PHQ9_item2=3", "patient_utterance_L008"
    ]
