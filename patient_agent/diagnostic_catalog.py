"""MentalKG label adapter for simulator fixtures; contains no medical rules."""
import json
from pathlib import Path

_root = Path(__file__).resolve().parents[2] / "external/MentalBench/resources/knowledge_graph/EN"
DISORDERS = json.loads((_root / "disorder.json").read_text(encoding="utf-8"))
SUPPORTED_DISORDERS = (*sorted(DISORDERS), "Insomnia")
_by_name = {record["name"]: code for code, record in DISORDERS.items()}
_aliases = {
    "mdd": ("Major Depressive Disorder",),
    "gad": ("Generalized Anxiety Disorder",),
    "ptsd": ("Posttraumatic Stress Disorder",),
    "ocd": ("Obsessive-Compulsive Disorder",),
    "bipolar": ("Bipolar I Disorder", "Bipolar II Disorder"),
}


def diagnostic_codes(value: str) -> list[str]:
    if value in SUPPORTED_DISORDERS:
        return [value]
    if value.lower() == "insomnia":
        return ["Insomnia"]
    return [_by_name[name] for name in _aliases[value.lower()]] if value.lower() in _aliases else [value]


def adapt_candidates(records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        for code in diagnostic_codes(record["code"]):
            result.append({**record, "code": code,
                           "label": record.get("name", ""),
                           "name": DISORDERS[code]["name"] if code in DISORDERS else "Insomnia Disorder",
                           "source": "simulation_fixture"})
    return result
