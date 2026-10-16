"""PA7：批量报告。

报告是给人看的，所以这里两头都验：汇总口径（层覆盖、错误类型、事实一致性、终止分布、
耗时）必须算对，Markdown 该有的段落和小节也必须出现。
"""

from __future__ import annotations

import json

from patient_agent.evaluator import ScenarioEvaluation
from patient_agent.report import (
    BatchReport,
    build_report,
    render_markdown,
    write_report,
)


def evaluation(
    scenario_id: str,
    *,
    ok: bool = True,
    layers: tuple[str, ...] = (),
    error_codes: tuple[str, ...] = (),
    fatal: tuple[str, ...] = (),
    violations: tuple[dict, ...] = (),
    termination: str | None = None,
    timing: dict | None = None,
) -> ScenarioEvaluation:
    return ScenarioEvaluation(
        scenario_id=scenario_id,
        ok=ok,
        layers_observed=list(layers),
        error_codes=list(error_codes),
        fatal_error_codes=list(fatal),
        guard_violations=list(violations),
        termination_observed=termination,
        timing=timing or {"count": 2, "total_ms": 10.0, "avg_ms": 5.0, "max_ms": 6.0},
    )


def sample() -> list[ScenarioEvaluation]:
    return [
        evaluation(
            "A",
            layers=("L1", "L2"),
            error_codes=("SAFETY_HOLD",),
            violations=({"code": "SEVERITY_INFLATION", "blocking": False},),
            termination=None,
            timing={"count": 4, "total_ms": 40.0, "avg_ms": 10.0, "max_ms": 20.0},
        ),
        evaluation(
            "B",
            ok=False,
            layers=("L1", "L3"),
            error_codes=("API_ERROR",),
            fatal=("API_ERROR",),
            violations=({"code": "FACT_MISMATCH", "blocking": False},),
            termination="dropout",
            timing={"count": 6, "total_ms": 60.0, "avg_ms": 10.0, "max_ms": 15.0},
        ),
    ]


# ------------------------------------------------------------------ 汇总口径
def test_pass_rate_and_counts() -> None:
    report = build_report(sample(), generated_at="2026-01-02T03:04:05")
    assert report.generated_at == "2026-01-02T03:04:05"
    assert (report.total, report.passed, report.failed) == (2, 1, 1)
    assert report.pass_rate == 0.5


def test_layer_coverage_counts_scenarios_per_layer() -> None:
    report = build_report(sample())
    assert report.layer_coverage == {"L1": 2, "L2": 1, "L3": 1}


def test_error_type_split_designed_vs_fatal() -> None:
    report = build_report(sample())
    assert report.error_code_counts == {"API_ERROR": 1, "SAFETY_HOLD": 1}
    assert report.fatal_error_counts == {"API_ERROR": 1}, "SAFETY_HOLD 是设计内路径"


def test_fact_consistency_only_counts_fact_codes() -> None:
    report = build_report(sample())
    # 严重度升级也属于事实一致性问题。
    assert report.guard_violation_counts == {"FACT_MISMATCH": 1, "SEVERITY_INFLATION": 1}
    assert report.fact_consistency_violations == 2


def test_termination_distribution_uses_placeholder() -> None:
    report = build_report(sample())
    assert report.termination_counts == {"未终止": 1, "dropout": 1}


def test_timing_is_aggregated_across_scenarios() -> None:
    report = build_report(sample())
    assert report.timing["turns"] == 10
    assert report.timing["total_ms"] == 100.0
    assert report.timing["avg_ms"] == 10.0
    assert report.timing["max_ms"] == 20.0


def test_empty_batch_is_safe() -> None:
    report = build_report([])
    assert report.total == 0 and report.pass_rate == 0.0
    assert report.layer_coverage == {}
    assert report.timing["turns"] == 0


# -------------------------------------------------------------------- Markdown
def test_markdown_has_all_sections() -> None:
    markdown = render_markdown(build_report(sample(), generated_at="2026-01-02T03:04:05"))
    assert "# Patient Agent 回放质量报告" in markdown
    assert "**1/2 通过**" in markdown
    assert "## 路径覆盖（每层覆盖场景数）" in markdown
    assert "| L1 | 2 |" in markdown
    assert "| L2 | 1 |" in markdown
    assert "| L5 | 0 |" in markdown
    assert "## 错误类型" in markdown
    assert "`API_ERROR` × 1" in markdown
    assert "其中非设计内故障" in markdown
    assert "## 事实一致性" in markdown
    assert "事实相关违规：2 处" in markdown
    assert "## 终止原因分布" in markdown
    assert "## 逐病例明细" in markdown
    assert "## 失败详情" in markdown
    assert "### B" in markdown


def test_markdown_clean_report_has_no_failure_section() -> None:
    clean = build_report([evaluation("A", layers=("L1",), termination="completed_improved")])
    markdown = render_markdown(clean)
    assert "## 失败详情" not in markdown
    assert "- 无" in markdown, "没有错误码时写明「无」"
    assert "- 未发现一致性违规" in markdown
    assert "completed_improved × 1" in markdown


def test_markdown_marks_failing_cases() -> None:
    markdown = render_markdown(build_report(sample()))
    assert "| A | 通过 |" in markdown
    assert "| B | 不通过 |" in markdown


def test_markdown_locates_guard_violations_by_turn() -> None:
    """非阻断违规也要能按轮次看到，否则「定位到具体轮次」只能靠翻 recorder。"""

    report = build_report(
        [
            evaluation(
                "A",
                violations=(
                    {
                        "code": "FACT_MISMATCH",
                        "message": "与病例事实不一致",
                        "blocking": False,
                        "turn_index": 7,
                    },
                ),
            )
        ]
    )
    markdown = render_markdown(report)
    assert "## 守卫违规明细（按轮次）" in markdown
    assert "| 第 7 轮 | `FACT_MISMATCH` | 否 | 与病例事实不一致 |" in markdown


def test_markdown_omits_violation_section_when_clean() -> None:
    markdown = render_markdown(build_report([evaluation("A", layers=("L1",))]))
    assert "## 守卫违规明细（按轮次）" not in markdown


# -------------------------------------------------------------------- 落盘
def test_write_report_roundtrip(tmp_path) -> None:
    report = build_report(sample(), generated_at="2026-01-02T03:04:05")
    json_path, md_path = write_report(report, tmp_path)

    assert json_path.name == "report.json"
    assert md_path.name == "report.md"
    assert json_path.is_file() and md_path.is_file()

    reloaded = BatchReport.model_validate(
        json.loads(json_path.read_text(encoding="utf-8"))
    )
    assert reloaded.model_dump() == report.model_dump()
    assert md_path.read_text(encoding="utf-8").startswith("# Patient Agent 回放质量报告")


def test_write_report_creates_nested_dir(tmp_path) -> None:
    json_path, md_path = write_report(build_report(sample()), tmp_path / "a" / "b")
    assert json_path.is_file() and md_path.is_file()
