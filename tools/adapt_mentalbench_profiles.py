"""Lossless MentalBench narrative -> L2/L3 shape, without diagnostic inference.

No network/model calls. Only question text enters the profile. Reference answers
and label-bearing source metadata are written separately. Missing clinical
extractions stay unknown; full diagnosis entry guards remain in force.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.layer2.adapters import L2ToL3Adapter
from layers.layer2.schemas import ClinicalIntake, DiagnosticContext, EvidenceNode, MotivationScores, ProfileFact
from layers.layer3.api import DiagnoseRequest
from shared.disorder_catalog import DISORDERS, SUPPORTED_DISORDERS, canonical_disorder
from shared.l2_l3_contract import missing_required_profile_fields, missing_required_profile_evidence


SOURCE_URL = "https://huggingface.co/datasets/hysong/MentalBench"
MODEL_FILES = {f"{prefix}main_{model}.json" for prefix in ("", "a_", "b_")
               for model in ("gpt5", "gemini", "qwen235")}


def reference_labels(record: dict[str, Any]) -> list[str]:
    """Decode the published answer using its option text, never a folder guess."""
    options = dict(re.findall(r"(?m)^\s*([A-D])\.\s*([^\r\n]+)", record["options"]))
    answer = record["answer"].strip()
    prefix, separator, names = answer.partition(".")
    if not separator or not re.fullmatch(r"[A-D](?:\s*(?:&|,|and)\s*[A-D])*", prefix):
        raise ValueError("Unrecognized answer notation")
    labels = []
    for letter in re.findall(r"[A-D]", prefix):
        label = canonical_disorder(options.get(letter, "").strip())
        if label not in SUPPORTED_DISORDERS:
            raise ValueError("Unmapped answer option")
        if DISORDERS.get(label, {}).get("name", "").casefold() not in names.casefold():
            raise ValueError("Answer letters and disease names disagree")
        if label not in labels:
            labels.append(label)
    return labels


def make_intake(narrative: str, case_id: str) -> ClinicalIntake:
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("Empty question narrative")
    # Paragraphs are quoted without paraphrase or invented symptom mappings.
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", narrative.strip()) if part.strip()]
    facts, nodes = [], []
    for index, quote in enumerate(paragraphs):
        evidence_id = f"{case_id}-p{index + 1:03d}"
        facts.append(ProfileFact(value=quote, sources=[evidence_id], patient_quotes=[quote]))
        nodes.append(EvidenceNode(
            evidence_id=evidence_id, source_type="benchmark_narrative",
            message_id=f"{case_id}-source", turn_index=1, quote=quote,
            target_path=f"key_narratives.{index}", created_by="offline_source_adapter",
        ))
    context = DiagnosticContext()
    return ClinicalIntake(
        session_id=case_id, current_stage="engaging", key_narratives=facts,
        motivation=MotivationScores(), diagnostic_context=context, evidence_nodes=nodes,
        information_sufficiency="insufficient", patient_confirmation="not_requested",
        missing_dimensions=[item.missing_label for item in missing_required_profile_fields({})],
        flags=["offline_benchmark_narrative", "clinical_extraction_pending", "not_patient_confirmed"],
    )


def adapt_record(record: dict[str, Any], case_id: str):
    # The profile builder receives no answer, options, feature seeds or prompts.
    intake = make_intake(record["question"], case_id)
    profile = L2ToL3Adapter.serialize_profile(intake)
    DiagnoseRequest.model_validate(profile)
    labels = reference_labels(record)
    missing = list(dict.fromkeys(item.field for item in [
        *missing_required_profile_fields(profile), *missing_required_profile_evidence(profile)
    ]))
    return intake, profile, labels, missing


def source_meta(relative: Path, source_id: str) -> dict[str, str]:
    parts = relative.parts
    level = parts[0]
    if level in {"low", "medium"} and len(parts) == 3:
        task_type = "type1" if level == "low" else "type2"
        code, variant = parts[1], "single"
    elif level == "high" and len(parts) == 5 and parts[3] in {"type3", "type4"}:
        task_type = parts[3]
        code = f"{parts[1]}-{parts[2]}"
        variant = "ambiguous" if task_type == "type3" else parts[-1].split("_", 1)[0]
    else:
        raise ValueError("Unrecognized source layout")
    filename = relative.stem
    model = filename.split("main_", 1)[-1]
    return {"source_file": relative.as_posix(), "source_case_id": source_id,
            "task_type": task_type, "source_code": code, "variant": variant, "generator": model,
            # Same ID across generators and paired type3/type4 forms stays
            # identifiable; seed-level grouping still needs later verification.
            "related_case_group": f"{code}/{source_id}"}


def convert(dataset_root: Path, output: Path, *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    paths = [path for path in sorted(dataset_root.rglob("*.json"))
             if "features" not in path.relative_to(dataset_root).parts and path.name in MODEL_FILES]
    if not paths:
        raise ValueError("No MentalBench case files found")
    names = ["profiles.jsonl", "l2_intakes.jsonl", "gold.jsonl", "sources.jsonl", "conversion_report.json"]
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in names):
        raise ValueError("Output already contains conversion files; choose a new folder to preserve frozen data")
    counts, generators, missing_counts = Counter(), Counter(), Counter()
    total, failures = 0, []
    handles = {name: (output / name).open("w", encoding="utf-8") for name in names[:-1]}
    def write(name, value):
        handles[name].write(json.dumps(value, ensure_ascii=False) + "\n")
    try:
        for path in paths:
            records = json.loads(path.read_text(encoding="utf-8-sig"))
            for source_id, record in records.items():
                if limit is not None and total >= limit:
                    break
                relative = path.relative_to(dataset_root)
                case_id = f"mb-{total + 1:06d}"
                try:
                    meta = source_meta(relative, source_id)
                    intake, profile, labels, missing = adapt_record(record, case_id)
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append({"source_file": relative.as_posix(), "source_case_id": source_id,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    continue
                total += 1
                counts[meta["task_type"]] += 1
                generators[meta["generator"]] += 1
                missing_counts.update(missing)
                write("profiles.jsonl", {"case_id": case_id, "profile": profile})
                write("l2_intakes.jsonl", {"case_id": case_id, "intake": intake.model_dump(mode="json")})
                write("gold.jsonl", {"case_id": case_id, "gold_labels": labels,
                                     "reference_answer": record["answer"],
                                     "reference_kind": "admissible_candidates" if meta["task_type"] == "type3" else "single_reference_label",
                                     "label_origin": "MentalBench_synthetic_graph_grounded",
                                     "source_url": SOURCE_URL, "source_edition": "DSM-5", **meta})
                write("sources.jsonl", {"case_id": case_id, "source_url": SOURCE_URL, **meta,
                                        "conversion_mode": "lossless_narrative_only",
                                        "semantic_extraction_completed": False,
                                        "patient_confirmation_completed": False,
                                        "full_l3_contract_ready": not missing, "missing_l3_fields": missing})
            if limit is not None and total >= limit:
                break
    finally:
        for handle in handles.values():
            handle.close()
    report = {"source_url": SOURCE_URL, "source_directory": str(dataset_root.resolve()),
              "conversion_mode": "lossless_narrative_only", "profile_schema": "L2ToL3Adapter.serialize_profile",
              "cases_converted": total, "by_task_type": dict(counts), "by_generator": dict(generators),
              "failures": failures, "missing_l3_field_counts": dict(missing_counts),
              "clinical_extraction_completed": False, "split_performed": False,
              "coverage": "MentalKG D001-D023; Insomnia not supplied by MentalBench",
              "runtime_knowledge_base_imported": False}
    (output / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "external/MentalBench/resources/dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/datasets/mentalbench_l2_format")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = convert(args.source, args.output, limit=args.limit)
    print(json.dumps({key: report[key] for key in ["cases_converted", "by_task_type", "by_generator", "failures"]}, ensure_ascii=True))
    if report["failures"]:
        raise SystemExit(1)
