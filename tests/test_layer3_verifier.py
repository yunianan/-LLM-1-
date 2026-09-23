from copy import deepcopy

from layers.layer3.verifier import IndependentVerifier


def _primary(candidate_confidence=0.99):
    met = {"A1", "A2", "A4", "A6", "A7"}
    return {
        "disorder": "MDD",
        "confidence": candidate_confidence,
        "dsm5_criteria_detail": [
            {
                "code": f"A{i}",
                "name": f"criterion-{i}",
                "status": "met" if f"A{i}" in met else "not_met",
                "met": f"A{i}" in met,
                "evidence_refs": [f"PHQ9_item{i}=3"] if f"A{i}" in met else [f"PHQ9_item{i}=0"],
            }
            for i in range(1, 10)
        ],
    }


def _complete_profile():
    return {
        "symptom_duration_days": 21,
        "functional_impairment": True,
        "mania_or_hypomania_history": False,
        "substance_or_medical_cause": False,
        "differential_diagnoses_assessed": ["Bipolar", "AdjustmentDisorder"],
        "evidence_conflicts": [],
    }


def test_verifier_score_is_independent_of_candidate_confidence():
    verifier = IndependentVerifier()
    low = verifier.verify(_primary(0.1), _complete_profile())
    high = verifier.verify(_primary(0.99), _complete_profile())

    assert low["score"] == high["score"] == 1.0
    assert low["routing"] == high["routing"] == "pass"


def test_verifier_returns_to_l2_when_exclusion_information_is_missing():
    profile = _complete_profile()
    del profile["mania_or_hypomania_history"]

    result = IndependentVerifier().verify(_primary(), profile)

    assert result["routing"] == "return_to_l2"
    assert "mania_or_hypomania_history" in {
        item["field"] for item in result["missing_dimensions"]
    }


def test_verifier_rejects_mdd_when_mania_history_is_present():
    profile = _complete_profile()
    profile["mania_or_hypomania_history"] = True

    result = IndependentVerifier().verify(_primary(), profile)

    assert result["routing"] == "reject"
    assert any("躁狂" in reason for reason in result["fail_reasons"])


def test_verifier_sends_conflicting_evidence_to_manual_review():
    profile = _complete_profile()
    profile["evidence_conflicts"] = [
        {"left": "PHQ9_item2=3", "right": "patient_denies_depressed_mood"}
    ]

    result = IndependentVerifier().verify(_primary(), profile)

    assert result["routing"] == "manual_review"


def test_verifier_rejects_high_confidence_candidate_with_missing_criterion_evidence():
    primary = deepcopy(_primary(0.99))
    primary["dsm5_criteria_detail"][0]["evidence_refs"] = []

    result = IndependentVerifier().verify(primary, _complete_profile())

    evidence_check = next(
        check for check in result["checks"] if check["check_id"] == "criteria_evidence"
    )
    assert evidence_check["status"] == "fail"
    assert result["score"] < 1.0
    assert result["routing"] == "reject"
