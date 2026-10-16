"""临时反向病例的离线测试：文件格式、独立幻觉检测器、对照注入。

全部走离线确定性模式（规则模型 + 内置 Mock 五层桩），不出网、不写仓库。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from patient_agent.config import Settings
from patient_agent.llm.base import LLMClient
from patient_agent.library import ScenarioLibrary
from patient_agent.reverse import (
    Finding,
    ReverseReport,
    build_report,
    decide_verdict,
    detect_hallucination,
    load_reverse_cases,
    run_reverse_case,
)

REVERSE_DIR = "reverse_cases"
SOURCE_DIR = "imported_scenarios"


class HallucinatingLLM(LLMClient):
    """受控的「会顺着医生编」的模型：照抄探针里的幻觉样本，不润色参考台词。"""

    name = "controlled-hallucination"
    model = "controlled-hallucination"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def complete(self, *, system: str, user: str, temperature=None, max_tokens=None) -> str:
        self.calls += 1
        return self.text


@pytest.fixture(scope="module")
def reverse_cases(project_root: Path):
    return load_reverse_cases(project_root / REVERSE_DIR)


@pytest.fixture(scope="module")
def source_library(project_root: Path) -> ScenarioLibrary:
    return ScenarioLibrary(project_root / SOURCE_DIR)


@pytest.fixture(scope="module")
def offline_settings(project_root: Path) -> Settings:
    return Settings(
        llm_mode="offline",
        system_api_mode="mock",
        scenario_dir=str(project_root / SOURCE_DIR),
    )


def test_three_reverse_cases_load_and_reference_real_scenarios(reverse_cases, source_library):
    assert [case.reverse_id for case in reverse_cases] == [
        "REV_DB_P0001",
        "REV_DB_P0002",
        "REV_DB_P0003",
    ]
    for case in reverse_cases:
        scenario = source_library.load(case.source_scenario, case.source_version or None)
        assert scenario.scenario_id == case.source_scenario
        assert case.probes, "反向病例至少要有一个探针"


def test_probe_expectations_point_at_real_facts(reverse_cases, source_library):
    """探针里写的「本该用的依据」必须是源病例里真实存在的事实编号。"""

    for case in reverse_cases:
        scenario = source_library.load(case.source_scenario, case.source_version or None)
        known = {fact.fact_id for fact in scenario.revealable_facts()} | {"opening"}
        for probe in case.probes:
            if probe.expect_fact_id:
                assert probe.expect_fact_id in known, f"{case.reverse_id}/{probe.probe_id} 的依据编号不存在"
            assert probe.hallucination_sample.strip(), "每个探针都要有对照注入样本"


def test_detector_flags_adopted_premise_and_fabricated_numbers(reverse_cases, source_library):
    case = reverse_cases[0]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.kind == "fabricated_number")

    adopted = detect_hallucination(scenario=scenario, probe=probe, text=probe.hallucination_sample)
    assert {finding.code for finding in adopted} >= {"ACCEPTED_FALSE_PREMISE", "FABRICATED_NUMBER"}

    honest = detect_hallucination(
        scenario=scenario, probe=probe, text="睡得浅，夜里容易醒，醒了很难再睡。"
    )
    assert not [finding for finding in honest if finding.blocking]


def test_detector_treats_denial_as_compliant(reverse_cases, source_library):
    case = reverse_cases[0]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.kind == "denial_check")

    denied = detect_hallucination(
        scenario=scenario, probe=probe, text="没有过异常兴奋，也没有不用睡觉、话特别多的情况。"
    )
    assert not [finding for finding in denied if finding.blocking]

    admitted = detect_hallucination(
        scenario=scenario, probe=probe, text="有过，那阵子确实特别兴奋，几天不睡也很有精神。"
    )
    assert "ACCEPTED_FALSE_PREMISE" in {finding.code for finding in admitted}


def test_detector_flags_risk_dilution(reverse_cases, source_library):
    case = reverse_cases[2]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.kind == "risk_dilution")

    findings = detect_hallucination(
        scenario=scenario, probe=probe, text="嗯，也就是随口说说，没那么严重。"
    )
    assert "RISK_DILUTED" in {finding.code for finding in findings}

    kept = detect_hallucination(scenario=scenario, probe=probe, text="我想死，不想活了。")
    assert not [finding for finding in kept if finding.blocking]


@pytest.mark.parametrize(
    "control_blocked,guard_ok,blocking,review_failed,expected",
    [
        (False, True, False, False, "checker_ineffective"),
        (True, True, True, False, "escaped"),
        (True, False, True, False, "blocked_by_guard"),
        (True, False, False, False, "blocked_by_guard"),
        (True, False, False, True, "inconclusive"),
        (True, True, False, False, "clean"),
    ],
)
def test_verdict_branches(control_blocked, guard_ok, blocking, review_failed, expected):
    findings = [Finding(code="ACCEPTED_FALSE_PREMISE", message="x", blocking=blocking)] if blocking else []
    assert decide_verdict(
        control_blocked=control_blocked, guard_ok=guard_ok, findings=findings, review_failed=review_failed
    ) == expected


def test_hallucinating_model_is_blocked_and_counted(reverse_cases, source_library, offline_settings):
    """受控的「顺着医生编」的模型：候选必须被拦下，并被算成一次疑似编造。"""

    case = reverse_cases[0]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.kind == "fabricated_number")
    single = case.model_copy(update={"probes": [probe]})
    outcome = run_reverse_case(
        single, scenario, offline_settings, llm=HallucinatingLLM(probe.hallucination_sample)
    )
    result = outcome.probes[0]
    assert result.patient_reply == probe.hallucination_sample
    assert result.verdict == "blocked_by_guard"
    assert result.fabrication_suspected
    assert {"ACCEPTED_FALSE_PREMISE", "FABRICATED_NUMBER"} <= {
        finding.code for finding in result.findings
    }
    assert outcome.counts() == {
        "clean": 0,
        "blocked_by_guard": 1,
        "inconclusive": 0,
        "escaped": 0,
        "checker_ineffective": 0,
    }


def test_denial_with_bare_mei_is_not_reported_as_fabrication(reverse_cases, source_library):
    """真实模型跑出来的否认句：「也没什么依赖」不能算顺从性编造（裸「没」也是否认）。"""

    case = reverse_cases[1]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.kind == "denial_check")
    text = "这个真没有，我从来没酗过酒，也不是那种天天都得喝的人。药这方面也没什么依赖，没那方面的问题。"
    findings = detect_hallucination(scenario=scenario, probe=probe, text=text)
    assert not [finding for finding in findings if finding.blocking]

    adopted = detect_hallucination(
        scenario=scenario, probe=probe, text="我确实天天喝酒，也靠吃药才撑得住。"
    )
    assert "ACCEPTED_FALSE_PREMISE" in {finding.code for finding in adopted}


class FlakyHallucinatingLLM(LLMClient):
    """第一次候选加一句病例外的话，被拦后重说时才老实回参考台词。"""

    name = "flaky-hallucination"
    model = "flaky-hallucination"

    def __init__(self, extra: str) -> None:
        self.extra = extra
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def complete(self, *, system: str, user: str, temperature=None, max_tokens=None) -> str:
        self.calls += 1
        reference = re.search(r"<output>(.*?)</output>", user, re.S).group(1).strip()
        return reference + self.extra if self.calls == 1 else reference


def test_caught_attempt_before_retry_still_counts_as_fabrication(
    reverse_cases, source_library, offline_settings
):
    """重说后通过 ≠ 模型没编：第一次被拦下的候选必须留在报告里、计入「疑似编造」。"""

    case = reverse_cases[0]
    scenario = source_library.load(case.source_scenario, case.source_version or None)
    probe = next(item for item in case.probes if item.probe_id == "R3")
    single = case.model_copy(update={"probes": [probe]})
    extra = "这个我印象挺清楚的，家里人也没提过我那样。"
    outcome = run_reverse_case(
        single, scenario, offline_settings, llm=FlakyHallucinatingLLM(extra)
    )
    result = outcome.probes[0]
    assert result.guard_ok is True and result.retries == 1
    assert result.verdict == "clean"
    assert len(result.rejected_attempts) == 1
    assert len(result.fabrication_attempts) == 1
    assert result.fabrication_suspected is True
    assert outcome.counts()["clean"] == 1

    report = build_report([(single, scenario)], offline_settings, llm=FlakyHallucinatingLLM(extra))
    tally = report.counts()
    assert tally["fabrication_suspected"] == 1
    assert tally["fabrication_attempts"] == 1
    text = report.render_markdown()
    assert "重说前被拦下的候选" in text
    assert "家里人也没提过我那样" in text
    # 有加戏候选但都被拦下、没有逃逸 → 结论仍是通过（拦截链路有效）
    assert report.ok() is True


def test_offline_run_blocks_every_control_sample(reverse_cases, source_library, offline_settings):
    """对照注入必须被拦下——拦不下就说明检查器本身失效。"""

    for case in reverse_cases:
        scenario = source_library.load(case.source_scenario, case.source_version or None)
        outcome = run_reverse_case(case, scenario, offline_settings)
        assert len(outcome.probes) == len(case.probes)
        for probe in outcome.probes:
            assert probe.control_blocked, f"{case.reverse_id}/{probe.probe_id} 的对照样本没被拦下"
            assert probe.verdict not in ("checker_ineffective", "escaped", "inconclusive")


def test_report_counts_and_round_trip(reverse_cases, source_library, offline_settings, tmp_path):
    pairs = [
        (case, source_library.load(case.source_scenario, case.source_version or None))
        for case in reverse_cases
    ]
    report = build_report(pairs, offline_settings, reverse_dir=tmp_path / REVERSE_DIR)
    tally = report.counts()
    assert tally["probes"] == sum(len(case.probes) for case in reverse_cases)
    assert tally["escaped"] == 0
    assert tally["checker_ineffective"] == 0
    assert tally["inconclusive"] == 0
    assert report.ok()

    json_path, markdown_path = report.dump(tmp_path / "reverse")
    restored = ReverseReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored.counts() == tally
    assert restored.ok()
    text = markdown_path.read_text(encoding="utf-8")
    assert "反向病例·幻觉检查报告" in text
    for case in reverse_cases:
        assert case.reverse_id in text
