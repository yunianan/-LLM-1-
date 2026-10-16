"""命令行入口。

只测「命令行契约」：子命令能不能跑、选项放前放后都认、输出目录真的落盘、
以及三种退出码（0 全通过 / 1 有病例没过 / 2 用法或配置错）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from patient_agent.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main


def run_cli(argv: list[str]) -> int:
    return main(argv)


# ---------------------------------------------------------------------- list
def test_list_prints_every_case(scenario_dir: Path, capsys) -> None:
    code = run_cli(["list", "--scenario-dir", str(scenario_dir)])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "共 6 例" in out
    for scenario_id in (
        "S01_low_risk_depression",
        "S02_insufficient_info",
        "S03_explicit_crisis",
        "S04_bipolar_exclusion",
        "S05_comorbid_anxiety_depression",
        "S06_treatment_dropout",
    ):
        assert scenario_id in out
    assert "v1.0.0" in out
    assert "标签：" in out


# ----------------------------------------------------------------------- run
def test_run_single_writes_recorder(tmp_path, scenario_dir, capsys) -> None:
    out_dir = tmp_path / "out"
    code = run_cli(
        [
            "run",
            "S03_explicit_crisis",
            "--scenario-dir",
            str(scenario_dir),
            "--out",
            str(out_dir),
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "S03_explicit_crisis" in out
    assert "[OK ]" in out or "[OK]" in out
    record_path = out_dir / "S03_explicit_crisis" / "recorder.json"
    assert record_path.is_file()
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    assert saved["scenario_id"] == "S03_explicit_crisis"
    assert saved["turns"]


def test_options_work_before_the_subcommand_too(tmp_path, scenario_dir) -> None:
    out_dir = tmp_path / "before"
    code = run_cli(
        ["--scenario-dir", str(scenario_dir), "--out", str(out_dir), "run", "S01_low_risk_depression"]
    )

    assert code == EXIT_OK
    assert (out_dir / "S01_low_risk_depression" / "recorder.json").is_file()


# ------------------------------------------------------------------- run-all
def test_run_all_writes_report_and_records(tmp_path, scenario_dir, capsys) -> None:
    out_dir = tmp_path / "runs"
    code = run_cli(["run-all", "--scenario-dir", str(scenario_dir), "--out", str(out_dir)])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "**6/6 通过**" in out
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "report.md").is_file()

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["total"] == 6 and report["failed"] == 0
    for scenario_id in report["layer_coverage"]:  # 顺手确认层都在报告里
        assert scenario_id.startswith("L")

    for scenario_id in (
        "S01_low_risk_depression",
        "S02_insufficient_info",
        "S03_explicit_crisis",
        "S04_bipolar_exclusion",
        "S05_comorbid_anxiety_depression",
        "S06_treatment_dropout",
    ):
        assert (out_dir / scenario_id / "recorder.json").is_file()


def test_run_all_only_subset_and_verbose(tmp_path, scenario_dir, capsys) -> None:
    out_dir = tmp_path / "subset"
    code = run_cli(
        [
            "run-all",
            "--only",
            "S01_low_risk_depression",
            "--scenario-dir",
            str(scenario_dir),
            "--out",
            str(out_dir),
            "--verbose",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "**1/1 通过**" in out
    assert "评测：通过" in out, "verbose 要打印逐病例明细"
    assert not (out_dir / "S03_explicit_crisis").exists()


# ------------------------------------------------------------------- 退出码 2
def test_unknown_scenario_is_usage_error(tmp_path, scenario_dir, capsys) -> None:
    code = run_cli(
        ["run", "S99_nope", "--scenario-dir", str(scenario_dir), "--out", str(tmp_path / "o")]
    )
    err = capsys.readouterr().err

    assert code == EXIT_USAGE
    assert "S99_nope" in err


def test_missing_scenario_dir_is_usage_error(tmp_path, capsys) -> None:
    code = run_cli(["list", "--scenario-dir", str(tmp_path / "not_here")])
    err = capsys.readouterr().err

    assert code == EXIT_USAGE
    assert "运行失败" in err


def test_no_subcommand_is_a_parser_error(scenario_dir) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["--scenario-dir", str(scenario_dir)])
    assert excinfo.value.code == 2


# ------------------------------------------------------------------- 退出码 1
def _broken_library(tmp_path: Path, scenario_dir: Path) -> Path:
    """造一个「必定不通过」的病例库：要求一个永远不会出现的事件。"""

    root = tmp_path / "scenarios"
    scenario_id = "S01_low_risk_depression"
    (root / scenario_id).mkdir(parents=True)
    raw = json.loads((scenario_dir / scenario_id / "1.0.0.json").read_text(encoding="utf-8"))
    raw["expected"]["events"].append(
        {"event_id": "never_happens", "layer": "L4", "required": True}
    )
    (root / scenario_id / "1.0.0.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "scenarios": [
                    {
                        "scenario_id": scenario_id,
                        "versions": ["1.0.0"],
                        "latest": "1.0.0",
                        "title": raw["title"],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def test_failing_case_exits_one(tmp_path, scenario_dir, capsys) -> None:
    root = _broken_library(tmp_path, scenario_dir)
    out_dir = tmp_path / "out"

    code = run_cli(["run", "S01_low_risk_depression", "--scenario-dir", str(root), "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert code == EXIT_FAILED
    assert "[FAIL]" in out
    assert "required_events" in out

    batch_out = tmp_path / "batch"
    code = run_cli(["run-all", "--scenario-dir", str(root), "--out", str(batch_out)])
    out = capsys.readouterr().out
    assert code == EXIT_FAILED
    assert "**0/1 通过**" in out
    assert "## 失败详情" in out
