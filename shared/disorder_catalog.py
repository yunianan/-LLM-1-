"""Project labels read directly from the unmodified MentalKG release.

Display groups and Chinese names are presentation metadata, not DSM rules.
Insomnia is the existing project extension and is not a MentalKG node.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any


MENTALKG_ROOT = Path(os.getenv("MENTALKG_PATH") or (
    Path(__file__).resolve().parents[2] / "external/MentalBench/resources/knowledge_graph/EN"
))
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1] / "knowledge/dsm5_supplement/EN"
SUPPLEMENT_REF = "knowledge/dsm5_supplement/EN"


def _read(relative: str) -> dict[str, Any]:
    return json.loads((MENTALKG_ROOT / relative).read_text(encoding="utf-8-sig"))


def _read_supplement(relative: str) -> dict[str, Any]:
    return json.loads((SUPPLEMENT_ROOT / relative).read_text(encoding="utf-8-sig"))


DISORDERS = _read("disorder.json")
if set(DISORDERS) != {f"D{i:03d}" for i in range(1, 24)}:
    raise ValueError("MentalKG must contain the original D001-D023 disorder mapping")
SUPPORTED_DISORDERS = (*sorted(DISORDERS), "Insomnia")
DISORDER_COUNT = len(SUPPORTED_DISORDERS)
DIAGNOSTIC_CRITERIA = _read("diagnostic_criteria.json")
SUPPLEMENT_DISORDERS = _read_supplement("disorder.json")
SUPPLEMENT_CRITERIA = _read_supplement("diagnostic_criteria.json")
MENTALKG_SYMPTOMS = {
    sid: record
    for path in sorted((MENTALKG_ROOT / "symptom").glob("*.json"))
    for sid, record in _read(f"symptom/{path.name}").items()
}
SUPPLEMENT_SYMPTOMS = _read_supplement("symptom/10insomnia.json")
if MENTALKG_SYMPTOMS.keys() & SUPPLEMENT_SYMPTOMS.keys():
    raise ValueError("Supplement symptom IDs must not overwrite MentalKG")
SYMPTOMS = {**MENTALKG_SYMPTOMS, **SUPPLEMENT_SYMPTOMS}

DISPLAY_GROUPS = {
    "attention": "注意缺陷与多动相关",
    "psychotic": "精神病性及相关障碍",
    "bipolar": "双相相关",
    "anxiety": "焦虑相关",
    "sleep": "睡眠相关",
    "depressive": "抑郁相关",
    "obsessive": "强迫及相关障碍",
    "stress": "创伤与应激相关",
    "eating": "进食相关",
}
_GROUP_NODES = {
    "attention": ("D001", "D002", "D003"),
    "psychotic": ("D004", "D005", "D006", "D007"),
    "bipolar": ("D008", "D009", "D010"),
    "anxiety": ("D011", "D012"),
    "sleep": ("Insomnia",),
    "depressive": ("D013", "D014", "D015"),
    "obsessive": ("D016", "D017"),
    "stress": ("D018", "D019", "D020"),
    "eating": ("D021", "D022", "D023"),
}
_GROUP_BY_NODE = {node: group for group, nodes in _GROUP_NODES.items() for node in nodes}
_ZH_NAMES = dict(zip(sorted(DISORDERS), (
    "注意缺陷/多动障碍（混合表现）", "注意缺陷/多动障碍（注意缺陷为主）",
    "注意缺陷/多动障碍（多动冲动为主）", "妄想障碍", "精神分裂症",
    "分裂情感性障碍（双相型）", "分裂情感性障碍（抑郁型）",
    "双相Ⅰ型障碍", "双相Ⅱ型障碍", "双相Ⅰ型障碍伴精神病性特征",
    "广泛性焦虑障碍", "特定恐怖症", "重性抑郁障碍", "持续性抑郁障碍",
    "重性抑郁障碍伴精神病性特征", "强迫症", "躯体变形障碍",
    "创伤后应激障碍", "急性应激障碍", "适应障碍", "神经性厌食",
    "神经性贪食", "暴食障碍",
)))
_ZH_NAMES["Insomnia"] = "失眠障碍"

# Only unambiguous old codes are accepted for historical report display.
# The former generic Bipolar label deliberately has no single-node mapping.
LEGACY_ALIASES = {"MDD": "D013", "GAD": "D011", "PTSD": "D018",
                  "OCD": "D016", "AdjustmentDisorder": "D020"}


def canonical_disorder(value: str) -> str:
    if value in SUPPORTED_DISORDERS:
        return value
    if value in LEGACY_ALIASES:
        return LEGACY_ALIASES[value]
    for node, record in DISORDERS.items():
        if value.casefold() == record["name"].casefold():
            return node
    if value.casefold() in {"insomnia disorder", "insomnia"}:
        return "Insomnia"
    return value


def disorder_metadata(value: str) -> dict[str, str]:
    node = canonical_disorder(value)
    if node not in SUPPORTED_DISORDERS:
        return {"disorder": value, "label": value, "display_name": value}
    group = _GROUP_BY_NODE[node]
    return {
        "disorder": node,
        "name": (DISORDERS if node in DISORDERS else SUPPLEMENT_DISORDERS)[node]["name"],
        "label": _ZH_NAMES[node],
        "display_group": group,
        "display_group_label": DISPLAY_GROUPS[group],
        "display_name": f"{DISPLAY_GROUPS[group]} · {_ZH_NAMES[node]}",
        "knowledge_source": ("MentalKG/DSM-5" if node in DISORDERS
                             else f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/{node}"),
    }


def grouped_catalog() -> list[dict[str, Any]]:
    return [{"group": group, "label": label,
             "disorders": [disorder_metadata(node) for node in _GROUP_NODES[group]]}
            for group, label in DISPLAY_GROUPS.items()]


def symptom_catalog() -> dict[str, str]:
    return {sid: record["description"] for sid, record in SYMPTOMS.items()}


@lru_cache
def candidate_knowledge(node: str) -> dict[str, Any]:
    if node == "Insomnia":
        root_ref = f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/Insomnia/required_criteria"
        return {
            "metadata": disorder_metadata(node),
            "source_ref": f"{SUPPLEMENT_REF}/diagnostic_criteria.json#/Insomnia",
            "criteria": SUPPLEMENT_CRITERIA[node],
            "symptoms": SUPPLEMENT_SYMPTOMS,
            "differential": _read_supplement("differential_diagnosis/Insomnia.json"),
            "events": _read_supplement("event.json"),
            "criterion_source_refs": {
                "A": root_ref,
                "A1": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S085",
                "A2": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S086",
                "A3": f"{SUPPLEMENT_REF}/symptom/10insomnia.json#/S087",
                "B": f"{root_ref}/additional_requirements/1",
                "C": f"{root_ref}/additional_requirements/2",
                "D": f"{root_ref}/min_duration",
                "E": f"{root_ref}/additional_requirements/3",
                "F": f"{root_ref}/additional_requirements/4",
                "G": f"{root_ref}/additional_requirements/5",
                "H": f"{root_ref}/additional_requirements/6",
            },
            "note": "DSM-5 项目补充，沿用 MentalKG 格式，非原图谱作者提供或审核；共存疾病不构成自动排除。",
        }
    if node not in DISORDERS:
        raise ValueError(f"Unsupported diagnostic candidate: {node}")
    criteria = DIAGNOSTIC_CRITERIA[node]
    ids: set[str] = set()
    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            ids.update(obj.get("symptom_pool", []))
            for v in obj.values(): collect(v)
        elif isinstance(obj, list):
            for v in obj: collect(v)
    collect(criteria)
    return {"metadata": disorder_metadata(node),
            "source_ref": f"MentalKG:diagnostic_criteria.json#/{node}",
            "criteria": criteria, "symptoms": {sid: SYMPTOMS[sid] for sid in sorted(ids)},
            "differential": _read(f"differential_diagnosis/{node}.json"),
            "events": _read("event.json")}


# Preserve the existing six treatment routes only. Other diagnoses have no
# validated project treatment mapping and are routed to manual review.
TREATMENT_KEY_BY_NODE = {"D008": "Bipolar", "D009": "Bipolar", "D011": "GAD",
                         "D013": "MDD", "D016": "OCD", "D018": "PTSD",
                         "D020": "AdjustmentDisorder", "Insomnia": "Insomnia"}


def treatment_key(node: str) -> str:
    return TREATMENT_KEY_BY_NODE.get(canonical_disorder(node), node)
