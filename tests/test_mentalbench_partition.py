from itertools import combinations

import pytest

from tools.mentalbench_partition import build_partition_plan, family_key, validate_partition


def fixture_rows():
    rows = []
    for index, pair in enumerate(combinations(["D001", "D002", "D003", "D004"], 2)):
        code = "-".join(pair)
        for variant in ("ambiguous", "a", "b"):
            rows.append({"case_id": f"case-{index}-{variant}", "source_code": code,
                         "related_case_group": f"{code}/h001", "gold_labels": list(pair)})
    return rows


def test_unordered_families_keep_reverse_directions_together():
    assert family_key("D005-D001") == family_key("D001-D005")
    assert family_key("D001") == "D001"
    with pytest.raises(ValueError):
        family_key("D001-D001")


def test_group_holdout_is_disjoint_repeatable_and_covers_labels():
    rows = fixture_rows()
    slots = {"single": (0, 0, 0), "pair_one_direction": (2, 2, 2), "pair_two_directions": (0, 0, 0)}
    plan, report = build_partition_plan(rows, slots=slots)
    assert build_partition_plan(list(reversed(rows)), slots=slots) == (plan, report)
    assert all(validate_partition(plan, rows).values())
    for part in ("kb", "dev", "eval"):
        labels = {label for row in rows if plan[family_key(row["source_code"])] == part for label in row["gold_labels"]}
        assert labels == {"D001", "D002", "D003", "D004"}


def test_metadata_rejects_cross_partition_source_group_reuse():
    rows = fixture_rows()
    rows[0]["related_case_group"] = rows[-1]["related_case_group"] = "shared-origin"
    plan = {family_key(row["source_code"]): "kb" if row is rows[0] else "eval" for row in rows}
    plan[family_key(rows[0]["source_code"])] = "kb"
    assert not all(validate_partition(plan, rows).values())
