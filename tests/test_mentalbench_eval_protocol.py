import json
import os

import pytest

from tools.mentalbench_eval_protocol import freeze, invalidate, score
from tools.mentalbench_seal import seal, unseal
from tools.prepare_mentalbench_split import file_digest, write_jsonl


pytestmark = pytest.mark.skipif(os.name != "nt", reason="CurrentUser DPAPI is Windows-only")


def test_seal_roundtrip_and_rejects_invalid_material():
    payload = {"gold": [{"case_id": "private", "gold_labels": ["D013"]}], "quarantine": ["不可公开"]}
    encrypted = seal(payload)
    assert unseal(encrypted) == payload
    assert b"private" not in encrypted
    with pytest.raises(ValueError, match="Unrecognized"):
        unseal(b"plaintext")
    with pytest.raises(OSError):
        unseal(encrypted[:-20])


def fixture_data(tmp_path):
    folder = tmp_path / "split"
    (folder / "engine").mkdir(parents=True)
    (folder / "sealed").mkdir()
    write_jsonl(folder / "engine/eval.inputs.jsonl", [{"case_id": "a", "profile": {}}])
    target = {"case_id": "a", "gold_labels": ["D013"], "reference_kind": "single_reference_label",
              "label_origin": "MentalBench_synthetic_graph_grounded", "source_code": "D013",
              "related_case_group": "D013/1", "task_type": "type1"}
    (folder / "sealed/eval.bundle.dpapi").write_bytes(seal({"gold": [target], "quarantine": []}))
    manifest = {p.relative_to(folder).as_posix(): file_digest(p) for p in folder.rglob("*") if p.is_file()}
    (folder / "integrity.json").write_text(json.dumps(manifest), encoding="utf-8")
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / "tools/example.py").write_text("VERSION = 1\n", encoding="utf-8")
    arm = {"id": "baseline", "reasoner_model": "fixture", "reasoner_version": "1",
           "verifier_model": "fixture", "verifier_version": "1", "prompt_version": "1",
           "tree_version": "1", "retrieval_policy": "disabled", "parameters": {"temperature": 0}}
    protocol = {"input_condition": "narrative_only", "entrypoint": "CLINICR.run",
                "primary_metrics": ["candidate_recall_micro"], "secondary_metrics": ["macro"],
                "top_k": 5, "bootstrap": 10, "seed": 20261015, "experiments": [arm]}
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"case_id": "a", "candidates": [{"disorder": "D013", "confidence": .9}]}])
    return folder, project, protocol, predictions


def run_declaration(folder, protocol, predictions):
    frozen = json.loads((folder / "protocol.frozen.json").read_text(encoding="utf-8"))
    return {"baseline": {"predictions": str(predictions), "declared_experiment": protocol["experiments"][0],
                         "code_snapshot_sha256": frozen["code_snapshot_sha256"]}}


def test_draft_cannot_freeze_and_registered_run_scores_once(tmp_path):
    folder, project, protocol, predictions = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="draft"):
        freeze(folder, {**protocol, "experiments": []}, project_root=project)
    assert not (folder / "evaluation_state.json").exists()
    freeze(folder, protocol, project_root=project)
    report = score(folder, run_declaration(folder, protocol, predictions))
    assert report["experiments"]["baseline"]["candidate_recall_micro"] == 1
    assert json.loads((folder / "evaluation_state.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="no second scoring"):
        score(folder, run_declaration(folder, protocol, predictions))


def test_incomplete_predictions_rejected_before_unsealing(tmp_path):
    folder, project, protocol, predictions = fixture_data(tmp_path)
    freeze(folder, protocol, project_root=project)
    write_jsonl(predictions, [])
    with pytest.raises(ValueError, match="entire frozen EVAL"):
        score(folder, run_declaration(folder, protocol, predictions))
    assert not (folder / "evaluation_reservation.json").exists()


def test_changed_experiment_or_data_rejected(tmp_path):
    folder, project, protocol, predictions = fixture_data(tmp_path)
    freeze(folder, protocol, project_root=project)
    run = run_declaration(folder, protocol, predictions)
    run["baseline"]["declared_experiment"] = {**protocol["experiments"][0], "tree_version": "changed"}
    with pytest.raises(ValueError, match="settings"):
        score(folder, run)
    (folder / "engine/eval.inputs.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared data changed"):
        score(folder, run_declaration(folder, protocol, predictions))


def test_invalidated_version_cannot_score(tmp_path):
    folder, project, protocol, predictions = fixture_data(tmp_path)
    freeze(folder, protocol, project_root=project)
    invalidate(folder, "Fixture leakage detected")
    with pytest.raises(ValueError, match="invalidated"):
        score(folder, run_declaration(folder, protocol, predictions))
