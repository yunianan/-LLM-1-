from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POSITIVE_LABELS = {
    "自杀未遂",
    "自杀准备行为",
    "自杀计划",
    "主动自杀意图",
    "被动自杀意图",
}


def load_cases(path: Path, limit: int | None) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        record
        for record in records
        if POSITIVE_LABELS.intersection(record["labels"])
    ]
    if limit is not None:
        cases = cases[:limit]
    return cases


def evaluate_one(record: dict) -> dict:
    from layers.layer1.risk_classifier import llm_classify

    started = time.perf_counter()
    try:
        result = llm_classify(record["text"], assessment_context={})
        analysis = result["analysis"]
        return {
            "idx": record["idx"],
            "labels": record["labels"],
            "detected": bool(analysis["suicide_signal_detected"]),
            "valid_output": True,
            "attempts": result["attempts"],
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error_code": None,
            "error_stage": None,
            "error_type": None,
        }
    except Exception as exc:
        return {
            "idx": record["idx"],
            "labels": record["labels"],
            "detected": False,
            "valid_output": False,
            "attempts": None,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error_code": getattr(exc, "code", type(exc).__name__),
            "error_stage": getattr(exc, "stage", None),
            "error_type": getattr(exc, "exception_type", type(exc).__name__),
        }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    detected = sum(item["detected"] for item in results)
    valid = sum(item["valid_output"] for item in results)
    per_label: dict[str, dict] = {}
    for label in sorted(POSITIVE_LABELS):
        rows = [item for item in results if label in item["labels"]]
        found = sum(item["detected"] for item in rows)
        per_label[label] = {
            "support": len(rows),
            "true_positive": found,
            "false_negative": len(rows) - found,
            "recall": round(found / len(rows), 6) if rows else None,
        }
    error_counts = Counter(
        item["error_code"] for item in results if not item["valid_output"]
    )
    latencies = [item["latency_seconds"] for item in results]
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "definition": "PsySUICIDE test split; five suicide-positive labels only",
        "total_positive_samples": total,
        "true_positive": detected,
        "false_negative": total - detected,
        "end_to_end_recall": round(detected / total, 6) if total else None,
        "valid_output_count": valid,
        "valid_output_rate": round(valid / total, 6) if total else None,
        "semantic_recall_among_valid_outputs": (
            round(detected / valid, 6) if valid else None
        ),
        "error_counts": dict(error_counts),
        "mean_latency_seconds": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "per_label": per_label,
        "false_negative_indices": [
            item["idx"] for item in results if not item["detected"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False, encoding="utf-8-sig")
    if not (os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")):
        raise RuntimeError("API key is not configured")

    cases = load_cases(args.dataset, args.limit)
    print(f"positive_cases={len(cases)} workers={args.workers}", flush=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, case): case for case in cases}
        for completed, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            results.append(item)
            if completed == 1 or completed % 10 == 0 or completed == len(cases):
                hits = sum(row["detected"] for row in results)
                errors = sum(not row["valid_output"] for row in results)
                print(
                    f"progress={completed}/{len(cases)} hits={hits} errors={errors}",
                    flush=True,
                )

    results.sort(key=lambda item: item["idx"])
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n",
        encoding="utf-8",
    )
    report = summarize(results)
    args.summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
