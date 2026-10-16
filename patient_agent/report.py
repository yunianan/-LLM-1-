"""PA7 批量报告：把多份评测结论汇总成「跑一遍质量报告」。

汇总口径按任务说明第七节的组织：路径覆盖、错误类型、事实一致性、模型耗时。

* 路径覆盖：每个场景走过哪些层，以及每层被覆盖了多少次；
* 错误类型：系统 API 错误码分布；
* 事实一致性：守卫违规按 code 分布，单列出 FACT_MISMATCH（与病例事实矛盾）；
* 模型耗时：逐轮耗时总/均/峰值，来自记录器的 ``timing_summary``。

报告同时输出 JSON（给机器）和 Markdown（给人看），落在 ``out_dir`` 下。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.evaluator import ScenarioEvaluation

#: 事实一致性关注的违规码：与病例既定事实/时间线直接矛盾。
FACT_CODES = ("FACT_MISMATCH", "FORBIDDEN_CLAIM", "TIMELINE_CONFLICT", "FABRICATED_NUMBER", "UNKNOWN_FACT", "DISCLOSURE_VIOLATION", "SEVERITY_INFLATION")


class BatchReport(BaseModel):
    """一次批量回放的质量报告。"""

    model_config = ConfigDict(extra="ignore")

    generated_at: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0

    # 每层被覆盖的场景数，例如 {"L1": 6, "L5": 5}。
    layer_coverage: dict[str, int] = Field(default_factory=dict)
    # 系统 API 错误码分布（含 SAFETY_HOLD 这类设计内的安全路径信号）。
    error_code_counts: dict[str, int] = Field(default_factory=dict)
    # 其中真正的故障（非设计内错误码）。
    fatal_error_counts: dict[str, int] = Field(default_factory=dict)
    # 守卫违规码分布。
    guard_violation_counts: dict[str, int] = Field(default_factory=dict)
    # 与事实一致性相关的违规轮数。
    fact_consistency_violations: int = 0
    # 终止原因分布。
    termination_counts: dict[str, int] = Field(default_factory=dict)
    # 全部场景合并的逐轮耗时统计（毫秒）。
    timing: dict[str, float] = Field(default_factory=dict)

    scenarios: list[ScenarioEvaluation] = Field(default_factory=list)


def _bump(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def build_report(
    evaluations: list[ScenarioEvaluation],
    *,
    generated_at: str | None = None,
) -> BatchReport:
    """把多份评测结论汇总成批量报告。"""

    passed = sum(1 for item in evaluations if item.ok)
    total = len(evaluations)

    layer_coverage: dict[str, int] = {}
    error_code_counts: dict[str, int] = {}
    fatal_error_counts: dict[str, int] = {}
    guard_violation_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    fact_violations = 0

    total_ms = 0.0
    max_ms = 0.0
    turn_count = 0

    for evaluation in evaluations:
        for layer in evaluation.layers_observed:
            _bump(layer_coverage, layer)
        for code in evaluation.error_codes:
            _bump(error_code_counts, code)
        for code in evaluation.fatal_error_codes:
            _bump(fatal_error_counts, code)
        for violation in evaluation.guard_violations:
            code = violation.get("code", "?")
            _bump(guard_violation_counts, code)
            if code in FACT_CODES:
                fact_violations += 1
        _bump(termination_counts, evaluation.termination_observed or "未终止")

        total_ms += float(evaluation.timing.get("total_ms", 0.0))
        max_ms = max(max_ms, float(evaluation.timing.get("max_ms", 0.0)))
        turn_count += int(evaluation.timing.get("count", 0))

    timing = {
        "turns": turn_count,
        "total_ms": round(total_ms, 3),
        "avg_ms": round(total_ms / turn_count, 3) if turn_count else 0.0,
        "max_ms": round(max_ms, 3),
    }

    return BatchReport(
        generated_at=generated_at or datetime.now().isoformat(timespec="seconds"),
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        layer_coverage=dict(sorted(layer_coverage.items())),
        error_code_counts=dict(sorted(error_code_counts.items())),
        fatal_error_counts=dict(sorted(fatal_error_counts.items())),
        guard_violation_counts=dict(sorted(guard_violation_counts.items())),
        fact_consistency_violations=fact_violations,
        termination_counts=dict(sorted(termination_counts.items())),
        timing=timing,
        scenarios=evaluations,
    )


def render_markdown(report: BatchReport) -> str:
    """把报告渲染成便于贴进文档的 Markdown。"""

    lines: list[str] = []
    lines.append("# Patient Agent 回放质量报告")
    lines.append("")
    lines.append(f"- 生成时间：{report.generated_at}")
    lines.append(
        f"- 结果：**{report.passed}/{report.total} 通过**（通过率 {report.pass_rate:.0%}）"
    )
    lines.append(
        f"- 模型耗时：合计 {report.timing.get('total_ms', 0.0)} ms / "
        f"{report.timing.get('turns', 0)} 轮，"
        f"均值 {report.timing.get('avg_ms', 0.0)} ms，"
        f"峰值 {report.timing.get('max_ms', 0.0)} ms"
    )
    lines.append("")

    lines.append("## 路径覆盖（每层覆盖场景数）")
    lines.append("")
    lines.append("| 层 | 覆盖场景数 |")
    lines.append("| --- | --- |")
    for layer in ("L1", "L2", "L3", "L4", "L5"):
        lines.append(f"| {layer} | {report.layer_coverage.get(layer, 0)} |")
    lines.append("")

    lines.append("## 错误类型")
    lines.append("")
    if report.error_code_counts:
        for code, count in report.error_code_counts.items():
            lines.append(f"- `{code}` × {count}")
    else:
        lines.append("- 无")
    if report.fatal_error_counts:
        lines.append("")
        lines.append(
            "其中非设计内故障："
            + "、".join(f"`{code}`×{n}" for code, n in report.fatal_error_counts.items())
        )
    lines.append("")

    lines.append("## 事实一致性")
    lines.append("")
    if report.guard_violation_counts:
        lines.append(f"- 事实相关违规：{report.fact_consistency_violations} 处")
        for code, count in report.guard_violation_counts.items():
            lines.append(f"- `{code}` × {count}")
    else:
        lines.append("- 未发现一致性违规")
    lines.append("")

    lines.append("## 终止原因分布")
    lines.append("")
    for reason, count in report.termination_counts.items():
        lines.append(f"- {reason} × {count}")
    lines.append("")

    lines.append("## 逐病例明细")
    lines.append("")
    lines.append("| 病例 | 结果 | 轮数 | 终止原因 | 覆盖层 | 失败项 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in report.scenarios:
        status = "通过" if item.ok else "不通过"
        failed = "、".join(check.name for check in item.failed_checks) or "-"
        layers = "、".join(item.layers_observed) or "-"
        lines.append(
            f"| {item.scenario_id} | {status} | {item.turn_count} | "
            f"{item.termination_observed or '未终止'} | {layers} | {failed} |"
        )
    lines.append("")

    failures = [item for item in report.scenarios if not item.ok]
    if failures:
        lines.append("## 失败详情")
        lines.append("")
        for item in failures:
            lines.append(f"### {item.scenario_id}")
            lines.append("")
            for check in item.failed_checks:
                lines.append(f"- **{check.name}**：{check.detail}")
            lines.append("")

    # 守卫违规即使不阻断（FACT_MISMATCH / SEVERITY_INFLATION）也要能看到，
    # 且必须带轮次，否则「失败可定位到具体轮次」这条验收标准落不了地。
    with_violations = [item for item in report.scenarios if item.guard_violations]
    if with_violations:
        lines.append("## 守卫违规明细（按轮次）")
        lines.append("")
        for item in with_violations:
            lines.append(f"### {item.scenario_id}")
            lines.append("")
            lines.append("| 轮次 | 错误码 | 阻断 | 说明 |")
            lines.append("| --- | --- | --- | --- |")
            for violation in item.guard_violations:
                turn_index = violation.get("turn_index")
                turn = f"第 {turn_index} 轮" if isinstance(turn_index, int) else "-"
                blocking = "是" if violation.get("blocking", True) else "否"
                message = str(violation.get("message", "")).replace("|", "\\|")
                lines.append(
                    f"| {turn} | `{violation.get('code', '?')}` | {blocking} | {message} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(report: BatchReport, out_dir: Path | str) -> tuple[Path, Path]:
    """把报告写盘，返回 (JSON 路径, Markdown 路径)。"""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    md_path = target / "report.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
