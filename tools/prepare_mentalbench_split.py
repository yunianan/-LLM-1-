"""Prepare versioned KB/DEV/EVAL data and seal evaluator labels, without LLM calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.disorder_catalog import DISORDERS
from tools.mentalbench_partition import PARTS, build_partition_plan, family_key, validate_partition
from tools.mentalbench_quality import RULE_VERSION, exclusion_reasons, screen_profile
from tools.mentalbench_seal import seal, unseal


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, output, seed=20261015):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Output version already exists; choose a new directory")
    profiles = read_jsonl(source / "profiles.jsonl")
    gold = read_jsonl(source / "gold.jsonl")
    by_id = {row["case_id"]: row for row in profiles}
    if len(by_id) != len(profiles) or set(by_id) != {row["case_id"] for row in gold}:
        raise ValueError("Profile/gold IDs are not uniquely aligned")
    plan, allocation = build_partition_plan(gold, seed=seed, expected_labels=set(DISORDERS))
    checks = validate_partition(plan, gold)
    active = {part: [] for part in PARTS}
    excluded = {part: [] for part in PARTS}
    review_counts = {part: Counter() for part in PARTS}
    exclusion_counts = {part: Counter() for part in PARTS}
    all_by_part = {part: [] for part in PARTS}
    for target in gold:
        part = plan[family_key(target["source_code"])]
        input_row = by_id[target["case_id"]]
        if set(input_row) != {"case_id", "profile"}:
            raise ValueError("Unexpected engine envelope fields")
        profile = input_row["profile"]
        forbidden = {"gold_labels", "reference_answer", "options", "source_code", "task_type", "variant",
                     "generator", "related_case_group", "source_file"}
        def leaked(value):
            if isinstance(value, dict):
                return bool(forbidden & set(value)) or any(leaked(v) for v in value.values())
            return isinstance(value, list) and any(leaked(v) for v in value)
        if leaked(profile):
            raise ValueError("Label-bearing metadata in profile")
        quality = screen_profile(profile)
        reasons = exclusion_reasons(quality, part)
        all_by_part[part].append(target)
        review_counts[part].update(quality["review_only"])
        item = {"input": input_row, "gold": target, "quality": quality, "reasons": reasons}
        if reasons:
            excluded[part].append(item)
            exclusion_counts[part].update(reasons)
        else:
            active[part].append(item)

    summary = {}
    for part in PARTS:
        rows = [item["gold"] for item in active[part]]
        labels = {label for row in rows for label in row["gold_labels"]}
        tasks = Counter(row["task_type"] for row in rows)
        models = Counter(row["generator"] for row in rows)
        total = len(rows)
        summary[part] = {"assigned": len(all_by_part[part]), "active": total, "quarantined": len(excluded[part]),
                         "source_codes": len({row["source_code"] for row in all_by_part[part]}),
                         "supergroups": len({family_key(row["source_code"]) for row in all_by_part[part]}),
                         "source_groups": len({row["related_case_group"] for row in all_by_part[part]}),
                         "label_coverage": len(labels), "task_counts": dict(tasks), "generator_counts": dict(models),
                         "task_proportions": {key: tasks[key] / total for key in ["type1", "type2", "type3", "type4"]},
                         "exclusion_reasons": dict(exclusion_counts[part]), "review_only_counts": dict(review_counts[part])}
        checks[f"label_coverage:{part}"] = labels == set(DISORDERS)
        proportions = [models[m] / total for m in ["gemini", "gpt5", "qwen235"]]
        checks[f"generator_balance:{part}"] = max(proportions) - min(proportions) <= .05
        checks[f"active_excludes_quarantine:{part}"] = not ({r["gold"]["case_id"] for r in active[part]} & {r["gold"]["case_id"] for r in excluded[part]})
    mix_difference = max(abs(summary["dev"]["task_proportions"][key] - summary["eval"]["task_proportions"][key]) for key in ["type1", "type2", "type3", "type4"])
    checks["task_type_mix_dev_eval"] = mix_difference <= .03
    checks["accounting_including_quarantine"] = sum(summary[p]["active"] + summary[p]["quarantined"] for p in PARTS) == len(gold)
    if not all(checks.values()):
        raise ValueError(f"Split validation failed: {[key for key, value in checks.items() if not value]}")

    # Seal before creating the output directory. There is no plaintext fallback.
    evaluation_bundle = {"gold": [item["gold"] for item in active["eval"]],
                         "quarantine": excluded["eval"], "assigned_gold": all_by_part["eval"],
                         "family_assignment": {key: part for key, part in plan.items() if part == "eval"}}
    encrypted = seal(evaluation_bundle)
    restored = unseal(encrypted)
    checks["sealed_material_roundtrip"] = restored == evaluation_bundle
    if not checks["sealed_material_roundtrip"]:
        raise ValueError("Sealed material verification failed")
    (output / "engine").mkdir(parents=True)
    (output / "development").mkdir()
    (output / "sealed").mkdir()
    for part in PARTS:
        write_jsonl(output / f"engine/{part}.inputs.jsonl", [item["input"] for item in active[part]])
    for part in ("kb", "dev"):
        write_jsonl(output / f"development/{part}.gold.jsonl", [item["gold"] for item in active[part]])
        write_jsonl(output / f"development/{part}.quarantine.jsonl", excluded[part])
    (output / "sealed/eval.bundle.dpapi").write_bytes(encrypted)
    report = {"version": "v2", "scope": "whole_L3_diagnostic_workflow", "formal_evaluation_run": False,
              "source_directory": str(source.resolve()), "allocation": allocation, "quality_rules": RULE_VERSION,
              "parts": summary, "task_mix_max_difference": mix_difference, "checks": checks,
              "all_checks_passed": all(checks.values()), "reference_labels": list(DISORDERS),
              "Insomnia": "system candidate retained; no gold coverage in MentalBench",
              "input_limitation": "narrative-only; clinical extraction and full diagnose() contract remain unresolved",
              "seal": "Windows CurrentUser DPAPI; same-account access remains possible; use separate evaluator account for stronger separation"}
    (output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digests = {path.relative_to(output).as_posix(): file_digest(path) for path in output.rglob("*") if path.is_file()}
    (output / "integrity.json").write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")
    protocol = {"status": "draft", "dataset_version": "v2", "scope": "whole_L3_diagnostic_workflow",
                "input_condition": "narrative_only", "entrypoint": "CLINICR.run",
                "top_k": 5, "bootstrap": 1000, "seed": seed,
                "primary_metrics": ["single_reference_label.candidate_recall_micro", "single_reference_label.candidate_ppv_micro",
                                    "admissible_candidates.candidate_recall_micro", "admissible_candidates.candidate_ppv_micro"],
                "secondary_metrics": ["macro", "top_k", "step2_false_elimination", "single_reference_final", "cluster_intervals"],
                "experiments": [], "notes": "Declare all model versions, prompts, retrieval and parameters before freezing; no EVAL has run."}
    (output / "protocol.draft.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/datasets/mentalbench_l2_format")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/datasets/mentalbench_split_v2")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.seed)
    print(json.dumps({"parts": report["parts"], "all_checks_passed": report["all_checks_passed"]}, ensure_ascii=True))
