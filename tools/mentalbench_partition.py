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
    family_labels = {key: {label for row in rows for label in row["gold_labels"]} for key, rows in grouped.items()}
    category_by_family = {key: category for category, keys in classes.items() for key in keys}
    rng = random.Random(seed)
    for attempt in range(200):
        plan = {}
        for category, counts in slots.items():
            keys = sorted(classes[category])
            rng.shuffle(keys)
            offset = 0
            for part, count in zip(PARTS, counts):
                plan.update((key, part) for key in keys[offset:offset + count])
                offset += count
        coverage = {part: Counter() for part in PARTS}
        for key, labels in family_labels.items():
            coverage[plan[key]].update(labels)
        for repair in range(500):
            missing = [(part, label) for part in PARTS for label in sorted(target_labels) if not coverage[part][label]]
            if not missing:
                return plan, {"seed": seed, "attempts": attempt + 1, "coverage_repairs": repair,
                              "algorithm": "fixed_slots_label_coverage_swaps_v2",
                              "experiment": "held_out_unordered_disorder_family",
                              "allocation_slots": slots,
                              "family_classes": {k: len(v) for k, v in classes.items()},
                              "record_counts_before_quarantine": dict(Counter(plan[family_key(row["source_code"])] for row in gold))}
            part, label = rng.choice(missing)
            swaps = []
            for incoming in sorted(plan):
                other = plan[incoming]
                if other == part or label not in family_labels[incoming]:
                    continue
                for outgoing in sorted(classes[category_by_family[incoming]]):
                    if plan[outgoing] != part:
                        continue
                    gain = 0
                    for destination, added, removed in ((part, incoming, outgoing), (other, outgoing, incoming)):
                        for item in family_labels[added] | family_labels[removed]:
                            before = coverage[destination][item]
                            after = before + (item in family_labels[added]) - (item in family_labels[removed])
                            gain += int(after > 0) - int(before > 0)
                    swaps.append((gain, incoming, outgoing))
            if not swaps:
                break
            best = max(value[0] for value in swaps)
            if best < 0:
                break
            _, incoming, outgoing = rng.choice([value for value in swaps if value[0] == best])
            other = plan[incoming]
            for destination, added, removed in ((part, incoming, outgoing), (other, outgoing, incoming)):
                coverage[destination].update(family_labels[added])
                coverage[destination].subtract(family_labels[removed])
            plan[incoming], plan[outgoing] = part, other
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
