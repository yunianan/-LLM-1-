"""PA2：版本化病例库。

病例库要能回答三个问题：有哪些病例、某病例有哪些版本、要哪个版本。
版本解析错了，评测会把「改了病例之后的期望」套到「改之前的运行」上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patient_agent.errors import ScenarioLibraryError, ScenarioValidationError
from patient_agent.library import ScenarioLibrary

EXPECTED_IDS = [
    "S01_low_risk_depression",
    "S02_insufficient_info",
    "S03_explicit_crisis",
    "S04_bipolar_exclusion",
    "S05_comorbid_anxiety_depression",
    "S06_treatment_dropout",
]


def test_index_lists_six_scenarios(library: ScenarioLibrary) -> None:
    assert library.ids() == EXPECTED_IDS


def test_entries_expose_versions_and_metadata(library: ScenarioLibrary) -> None:
    entries = {entry.scenario_id: entry for entry in library.entries()}
    for scenario_id in EXPECTED_IDS:
        entry = entries[scenario_id]
        assert entry.versions == ["1.0.0"]
        assert entry.latest == "1.0.0"
        assert entry.title
        assert entry.tags


def test_latest_version_is_resolved_when_unspecified(library: ScenarioLibrary) -> None:
    assert library.entry("S01_low_risk_depression").resolve_version(None) == "1.0.0"


def test_unknown_scenario_raises_with_hint(library: ScenarioLibrary) -> None:
    with pytest.raises(ScenarioLibraryError) as excinfo:
        library.load("S99_not_there")
    assert "S99_not_there" in str(excinfo.value)
    # 报错要顺手告诉调用方有哪些可用 ID，否则排查很费劲。
    assert "S01_low_risk_depression" in str(excinfo.value)


def test_unregistered_version_is_rejected(library: ScenarioLibrary) -> None:
    with pytest.raises(ScenarioLibraryError):
        library.load("S01_low_risk_depression", "9.9.9")


def test_path_layout_is_scenario_id_over_version(scenario_dir: Path) -> None:
    library = ScenarioLibrary(scenario_dir)
    path = library.path_for("S01_low_risk_depression", "1.0.0")
    assert path == scenario_dir / "S01_low_risk_depression" / "1.0.0.json"
    assert path.is_file()


def test_load_is_cached(scenario_dir: Path) -> None:
    library = ScenarioLibrary(scenario_dir)
    first = library.load("S01_low_risk_depression")
    second = library.load("S01_low_risk_depression")
    assert first is second


def test_load_all_returns_every_scenario(library: ScenarioLibrary) -> None:
    loaded = library.load_all()
    assert [item.scenario_id for item in loaded] == EXPECTED_IDS


def test_seed_is_declared_per_scenario(all_scenarios) -> None:
    """每例都要有明确 seed，否则随机分支无法复现。"""

    seeds = {item.seed for item in all_scenarios.values()}
    assert len(seeds) == len(all_scenarios)
    assert all(isinstance(item.seed, int) for item in all_scenarios.values())


def test_malformed_scenario_file_is_reported(tmp_path: Path) -> None:
    """坏病例要报 ScenarioValidationError，而不是静默跑出一个假结论。"""

    root = tmp_path / "scenarios"
    (root / "SX_broken").mkdir(parents=True)
    (root / "SX_broken" / "1.0.0.json").write_text(
        json.dumps({"scenario_id": "SX_broken"}), encoding="utf-8"
    )
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "scenarios": [
                    {
                        "scenario_id": "SX_broken",
                        "versions": ["1.0.0"],
                        "latest": "1.0.0",
                        "title": "缺字段",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    library = ScenarioLibrary(root)
    with pytest.raises(ScenarioValidationError):
        library.load("SX_broken")
