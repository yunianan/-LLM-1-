import json
import pytest
from tools.prepare_dsm5_case_split import prepare, write_split
from tools.evaluate_candidate_predictions import evaluate


def record(case_id, label="D013", category="depressive"):
    # Synthetic mechanics fixture: never used as a clinical reference case.
    return {"case_id": case_id, "source_edition": "DSM-5", "source_ref": "engineering-fixture",
            "author_primary_diagnosis": "fixture", "mapping_reviewed": True,
            "input_reviewed_no_diagnosis_leak": True, "gold_labels": [label], "primary_label": label,
            "primary_dsm_category": category, "vignette": f"synthetic mechanics {case_id}",
            "profile": {"key_narratives": [case_id]}}


def test_stratified_split_keeps_cases_disjoint_and_labels_out_of_input(tmp_path):
    records = [record(f"m{i}") for i in range(5)] + [record(f"a{i}", "D011", "anxiety") for i in range(5)]
    dev, test, report = prepare(records)
    assert len(dev) == 4 and len(test) == 6
    assert not {r["case_id"] for r in dev} & {r["case_id"] for r in test}
    assert {r["primary_dsm_category"] for r in test} == {"depressive", "anxiety"}
    assert prepare(list(reversed(records))) == (dev, test, report)
    write_split(tmp_path, dev, test, report)
    inputs = [json.loads(line) for line in (tmp_path / "test.inputs.jsonl").read_text().splitlines()]
    assert all(set(row) == {"case_id", "profile"} for row in inputs)
    assert "Insomnia" in report["untested_labels"]


def test_exclusions_are_explicit_and_singleton_not_claimed_tested():
    rows = [record("one"), record("tr"), record("unclear"), record("wrong")]
    rows[1]["source_edition"] = "DSM-5-TR"
    rows[2]["mapping_reviewed"] = False
    rows[3]["primary_dsm_category"] = "ui_group"
    dev, test, report = prepare(rows)
    assert len(dev) == 1 and not test
    assert len(report["excluded"]) == 3
    assert report["singleton_categories_dev_only"] == ["depressive"]


def test_duplicate_story_cannot_leak_between_partitions():
    first, second = record("a"), record("b")
    second["vignette"] = first["vignette"]
    with pytest.raises(ValueError, match="Duplicate vignette"):
        prepare([first, second])


def test_candidate_metrics_count_false_positives_and_empty_cases():
    gold = [{"case_id": "a", "gold_labels": ["D013", "Insomnia"]},
            {"case_id": "b", "gold_labels": ["D011"]}]
    predictions = [{"case_id": "a", "candidates": [
        {"disorder": "D013", "confidence": 0.8}, {"disorder": "D009", "confidence": 0.9},
        {"disorder": "Insomnia", "confidence": 0.7}]}, {"case_id": "b", "candidates": []}]
    result = evaluate(gold, predictions, 1)
    assert result["candidate_ppv_micro"] == 2 / 3
    assert result["candidate_recall_micro"] == 2 / 3
    assert result["top_k_recall_micro"] == 0
    assert result["empty_candidate_rate"] == 0.5
    assert result["per_label"]["D001"]["recall"] is None
    with pytest.raises(ValueError, match="exactly match"):
        evaluate(gold, predictions[:1])
