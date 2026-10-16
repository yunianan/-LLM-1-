"""Freeze a preregistered L3 experiment and score sealed EVAL exactly once.

No diagnostic model calls. This is a local workflow guard, not protection
against an operator who controls the same Windows account and all files.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_candidate_predictions import evaluate
from tools.prepare_mentalbench_split import file_digest, read_jsonl
from tools.mentalbench_seal import unseal


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_integrity(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "integrity.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = (folder / relative).resolve()
        if folder not in path.parents or not path.is_file() or file_digest(path) != expected:
            raise ValueError("Prepared data changed or invalid integrity path; create a new version")


def validate_protocol(protocol):
    if protocol.get("input_condition") not in {"narrative_only", "reviewed_structured_profile"}:
        raise ValueError("Declare the input condition")
    if not protocol.get("entrypoint") or not protocol.get("primary_metrics") or not protocol.get("secondary_metrics"):
        raise ValueError("Declare entrypoint and metrics before EVAL")
    if not isinstance(protocol.get("top_k"), int) or protocol["top_k"] < 1:
        raise ValueError("Declare a positive Top-K")
    if not isinstance(protocol.get("bootstrap"), int) or protocol["bootstrap"] < 1:
        raise ValueError("Declare bootstrap replicates")
    if not isinstance(protocol.get("seed"), int):
        raise ValueError("Declare the random seed")
    arms = protocol.get("experiments")
    if not isinstance(arms, list) or not arms:
        raise ValueError("Declare all comparison experiments; the draft cannot unlock EVAL")
    ids = set()
    fields = ["id", "reasoner_model", "reasoner_version", "verifier_model", "verifier_version",
              "prompt_version", "tree_version", "retrieval_policy", "parameters"]
    for arm in arms:
        for field in fields:
            value = arm.get(field)
            if not value or isinstance(value, str) and value.casefold() in {"latest", "default", "todo", "tbd"}:
                raise ValueError(f"Undeclared experiment setting: {field}")
        if arm["id"] in ids or not isinstance(arm["parameters"], dict):
            raise ValueError("Experiment IDs must be unique and parameters explicit")
        ids.add(arm["id"])


def freeze(folder, protocol, *, project_root=ROOT):
    folder, project_root = Path(folder), Path(project_root)
    if (folder / "evaluation_state.json").exists():
        raise ValueError("This version already has an evaluation state")
    validate_protocol(protocol)
    verify_integrity(folder)
    # Capture current working files rather than pretending an uncommitted tree
    # is represented by HEAD. No .env, keys, patient traces or dependency trees.
    archive = folder / "code_snapshot.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for directory in ("layers", "shared", "knowledge", "tools"):
            for path in sorted((project_root / directory).rglob("*")):
                if path.is_file() and path.suffix in {".py", ".json"}:
                    bundle.write(path, "project/" + path.relative_to(project_root).as_posix())
        if (project_root / "pyproject.toml").is_file():
            bundle.write(project_root / "pyproject.toml", "project/pyproject.toml")
        upstream = project_root.parent / "external/MentalBench/resources/knowledge_graph/EN"
        for path in sorted(upstream.rglob("*.json")):
            bundle.write(path, "external/MentalBench/resources/knowledge_graph/EN/" + path.relative_to(upstream).as_posix())
    frozen = {**protocol, "status": "frozen", "frozen_at": datetime.now(timezone.utc).isoformat(),
              "code_snapshot_sha256": file_digest(archive),
              "integrity_manifest_sha256": file_digest(folder / "integrity.json"),
              "formal_evaluation_completed": False}
    write_json(folder / "protocol.frozen.json", frozen)
    state = {"status": "frozen", "protocol_sha256": file_digest(folder / "protocol.frozen.json"),
             "formal_evaluation_completed": False, "prediction_selection_after_results_allowed": False}
    write_json(folder / "evaluation_state.json", state)
    return state


def score(folder, runs):
    folder = Path(folder)
    state = json.loads((folder / "evaluation_state.json").read_text(encoding="utf-8"))
    if state["status"] != "frozen" or (folder / "evaluation_reservation.json").exists():
        raise ValueError("EVAL has been reserved, completed or invalidated; no second scoring")
    if file_digest(folder / "protocol.frozen.json") != state["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    protocol = json.loads((folder / "protocol.frozen.json").read_text(encoding="utf-8"))
    verify_integrity(folder)
    if file_digest(folder / "code_snapshot.zip") != protocol["code_snapshot_sha256"] or file_digest(folder / "integrity.json") != protocol["integrity_manifest_sha256"]:
        raise ValueError("Frozen snapshot or integrity manifest changed")
    arms = {arm["id"]: arm for arm in protocol["experiments"]}
    if set(runs) != set(arms):
        raise ValueError("Provide exactly all preregistered experiments together")
    expected_ids = {row["case_id"] for row in read_jsonl(folder / "engine/eval.inputs.jsonl")}
    outputs = {}
    for arm_id, run in runs.items():
        if run.get("declared_experiment") != arms[arm_id] or run.get("code_snapshot_sha256") != protocol["code_snapshot_sha256"]:
            raise ValueError("Prediction run settings do not match the frozen experiment")
        predictions = read_jsonl(run["predictions"])
        if len(predictions) != len(expected_ids) or {row["case_id"] for row in predictions} != expected_ids:
            raise ValueError("Predictions must cover the entire frozen EVAL input exactly once")
        outputs[arm_id] = predictions
    # Reserve before opening labels. A scoring failure cannot be used as an
    # unlogged opportunity to see labels and try another set of predictions.
    reservation = {"status": "reserved", "started_at": datetime.now(timezone.utc).isoformat(),
                   "runs": {key: {"prediction_sha256": file_digest(value["predictions"])} for key, value in runs.items()}}
    with (folder / "evaluation_reservation.json").open("x", encoding="utf-8") as stream:
        json.dump(reservation, stream, indent=2)
    try:
        material = unseal((folder / "sealed/eval.bundle.dpapi").read_bytes())
        results = {arm_id: evaluate(material["gold"], predictions, protocol["top_k"],
                                   bootstrap=protocol["bootstrap"], seed=protocol["seed"])
                   for arm_id, predictions in outputs.items()}
        report = {"protocol_sha256": state["protocol_sha256"], "experiments": results,
                  "completed_at": datetime.now(timezone.utc).isoformat(), "input_condition": protocol["input_condition"]}
        write_json(folder / "evaluation_report.json", report)
        state.update(status="completed", formal_evaluation_completed=True)
        write_json(folder / "evaluation_state.json", state)
        return report
    except Exception as exc:
        state.update(status="failed_after_unseal", failure_type=type(exc).__name__)
        write_json(folder / "evaluation_state.json", state)
        raise


def invalidate(folder, reason):
    folder = Path(folder)
    if not reason.strip():
        raise ValueError("An invalidation reason is required")
    state = json.loads((folder / "evaluation_state.json").read_text(encoding="utf-8"))
    state.update(status="invalidated", reason=reason, invalidated_at=datetime.now(timezone.utc).isoformat())
    write_json(folder / "evaluation_state.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "score", "invalidate"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.action == "freeze":
        if args.protocol is None: parser.error("--protocol is required")
        result = freeze(args.folder, json.loads(args.protocol.read_text(encoding="utf-8")))
    elif args.action == "score":
        if args.runs is None: parser.error("--runs is required")
        result = score(args.folder, json.loads(args.runs.read_text(encoding="utf-8")))
        result = {"formal_evaluation_completed": True, "experiments": list(result["experiments"])}
    else:
        result = invalidate(args.folder, args.reason)
    print(json.dumps(result, ensure_ascii=True))
