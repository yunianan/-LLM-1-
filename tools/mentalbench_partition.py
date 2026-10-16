"""Versioned MentalBench partitions for the whole L3 diagnostic workflow.

An unordered-pair holdout is a deliberately strict generalization experiment,
not a claim that sharing a diagnosis between independent cases is data leakage.
"""
from collections import Counter, defaultdict
import random


PARTS = ("kb", "dev", "eval")
DEFAULT_SLOTS = {"single": (11, 5, 7), "pair_one_direction": (24, 10, 17),
                 "pair_two_directions": (10, 3, 5)}


def family_key(source_code):
    nodes = source_code.split("-")
    if len(nodes) not in (1, 2) or any(len(n) != 4 or n[0] != "D" or not n[1:].isdigit() for n in nodes):
        raise ValueError("Invalid source_code")
    if len(set(nodes)) != len(nodes):
        raise ValueError("A differential pair requires distinct disorders")
    return "-".join(sorted(nodes))


def build_partition_plan(gold, *, seed=20261015, slots=None, expected_labels=None):
    slots = slots or DEFAULT_SLOTS
    grouped, seen = defaultdict(list), set()
    for row in gold:
        if row["case_id"] in seen:
            raise ValueError("Duplicate case_id")
        seen.add(row["case_id"])
        grouped[family_key(row["source_code"])].append(row)
    target_labels = set(expected_labels or {label for row in gold for label in row["gold_labels"]})
    classes = {key: [] for key in slots}
    for key, rows in grouped.items():
        directions = {row["source_code"] for row in rows}
        category = "single" if "-" not in key else "pair_one_direction" if len(directions) == 1 else "pair_two_directions"
        classes[category].append(key)
    for category, counts in slots.items():
        if len(counts) != 3 or len(classes[category]) != sum(counts):
            raise ValueError(f"Source family count does not match declared policy: {category}")
    rng = random.Random(seed)
    for attempt in range(20000):
        plan = {}
        for category, counts in slots.items():
            keys = sorted(classes[category])
            rng.shuffle(keys)
            offset = 0
            for part, count in zip(PARTS, counts):
                plan.update((key, part) for key in keys[offset:offset + count])
                offset += count
        coverage = {part: set() for part in PARTS}
        for key, rows in grouped.items():
            coverage[plan[key]].update(label for row in rows for label in row["gold_labels"])
        if all(coverage[part] == target_labels for part in PARTS):
            return plan, {"seed": seed, "attempts": attempt + 1,
                          "experiment": "held_out_unordered_disorder_family",
                          "allocation_slots": slots,
                          "family_classes": {k: len(v) for k, v in classes.items()},
                          "record_counts_before_quarantine": dict(Counter(plan[family_key(row["source_code"])] for row in gold))}
    raise ValueError("No partition met label coverage under the declared policy")


def validate_partition(plan, gold):
    """Check case/source/related-group/family disjointness without case text."""
    buckets = {part: {"case_id": set(), "source_code": set(), "related_case_group": set(), "family": set()}
               for part in PARTS}
    for row in gold:
        family = family_key(row["source_code"])
        part = plan[family]
        for field in ("case_id", "source_code", "related_case_group"):
            buckets[part][field].add(row[field])
        buckets[part]["family"].add(family)
    checks = {}
    for i, first in enumerate(PARTS):
        for second in PARTS[i + 1:]:
            for field in buckets[first]:
                checks[f"{field}_disjoint:{first}:{second}"] = not bool(buckets[first][field] & buckets[second][field])
    checks["record_accounting"] = sum(len(buckets[p]["case_id"]) for p in PARTS) == len(gold)
    return checks
