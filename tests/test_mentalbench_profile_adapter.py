import json
from pathlib import Path

import pytest

from layers.layer2.adapters import L2ToL3Adapter
from layers.layer2.guards import L2ExitGuardError
from layers.layer2.schemas import ClinicalIntake
from layers.layer3.api import DiagnoseRequest, build_input_contract_report
from layers.layer3.diagnosis.clinicr import CLINICR
from tools.adapt_mentalbench_profiles import adapt_record, convert, reference_labels


def record(answer="A. Major Depressive Disorder"):
    return {"question": "My sleep and work have changed.\n\nI need more time to explain what happened.",
            "options": "A. Major Depressive Disorder\nB. Bipolar I Disorder\nC. Generalized Anxiety Disorder\nD. Specific Phobia",
            "answer": answer,
            "features": {"disease_code": "D013", "duration": "100 years", "symptoms": ["S024"]},
            "prompt": "THIS IS HIDDEN GENERATION MATERIAL"}


def test_profile_uses_real_l2_serialization_without_inventing_collected_fields():
    intake, profile, labels, missing = adapt_record(record(), "mb-000001")
    ClinicalIntake.model_validate(intake.model_dump(mode="json"))
    DiagnoseRequest.model_validate(profile)
    assert profile == L2ToL3Adapter.serialize_profile(intake)
    assert labels == ["D013"]
    assert profile["key_symptoms_extracted"] == []
    assert profile["symptom_duration_days"] is None
    assert profile["functional_impairment"] is None
    assert profile["mania_or_hypomania_history"] is None
    assert profile["evidence_conflicts"] is None
    assert profile["motivation"]["confidence"] is None
    assert not any(key.startswith("PHQ") for key in profile)
    assert len(missing) == 5


def test_answers_options_and_hidden_features_do_not_enter_profile():
    first = record()
    second = record("B. Bipolar I Disorder")
    second["features"] = {"disease_code": "D008", "duration": "1 day"}
    second["prompt"] = "completely different hidden generation prompt"
    _, a, gold_a, _ = adapt_record(first, "mb-000001")
    _, b, gold_b, _ = adapt_record(second, "mb-000001")
    assert a == b
    assert gold_a != gold_b
    encoded = json.dumps(a)
    assert "D013" not in encoded and "D008" not in encoded
    assert "GENERATION" not in encoded and "100 years" not in encoded


def test_evidence_quotes_and_reference_whitelist_are_traceable():
    _, profile, _, _ = adapt_record(record(), "mb-000001")
    known = CLINICR._known_evidence_refs(profile)
    for index, fact in enumerate(profile["key_narratives"]):
        node = profile["evidence_nodes"][index]
        assert node["quote"] in record()["question"]
        assert fact["sources"] == [node["evidence_id"]]
        assert node["evidence_id"] in known
        assert node["source_type"] == "benchmark_narrative"
        assert node["created_by"] == "offline_source_adapter"


def test_offline_serialization_does_not_grant_normal_l2_or_full_diagnosis_entry():
    intake, profile, _, _ = adapt_record(record(), "mb-000001")
    assert intake.patient_confirmation == "not_requested"
    with pytest.raises(L2ExitGuardError):
        L2ToL3Adapter().to_profile(intake)
    assert build_input_contract_report(profile)["verifier_routing"] == "return_to_l2"


def test_ambiguous_reference_remains_a_candidate_set():
    assert reference_labels(record("A & B. Major Depressive Disorder & Bipolar I Disorder")) == ["D013", "D008"]
    with pytest.raises(ValueError):
        reference_labels(record("A. Bipolar I Disorder"))


def test_conversion_skips_feature_copies_and_preserves_related_case_metadata(tmp_path):
    source = tmp_path / "source"
    for relative in ["high/D013/D008/type3/main_gpt5.json", "high/D013/D008/type4/a_main_gemini.json",
                     "high/D013/D008/features/main_qwen235.json"]:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"D013-D008_h001": record()}), encoding="utf-8")
    output = tmp_path / "converted"
    report = convert(source, output)
    assert report["cases_converted"] == 2
    assert not report["failures"]
    inputs = [json.loads(line) for line in (output / "profiles.jsonl").read_text().splitlines()]
    gold = [json.loads(line) for line in (output / "gold.jsonl").read_text().splitlines()]
    assert all(set(row) == {"case_id", "profile"} for row in inputs)
    assert len({row["related_case_group"] for row in gold}) == 1
    assert {row["case_id"] for row in inputs} == {row["case_id"] for row in gold}
    with pytest.raises(ValueError, match="already contains"):
        convert(source, output)


def test_bad_reference_is_reported_instead_of_guessed_from_disease_folder(tmp_path):
    path = tmp_path / "source/low/D013/main_gpt5.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"D013_l001": record("unknown")}), encoding="utf-8")
    report = convert(tmp_path / "source", tmp_path / "output")
    assert report["cases_converted"] == 0
    assert len(report["failures"]) == 1
