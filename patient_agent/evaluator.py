"""PA7 评测器：把一次运行的记录对照病例的 ``expected`` 判分。

评测器只读记录、不跑会话，因此「跑一次、评多次」是廉价的，也方便回归比对。

判分口径（与任务说明第七节的自检清单对应）：

* ``required_events``：病例声明的必需事件是否都观测到；
* ``forbidden_events``：一旦出现即判该次评测不通过（路径走错）；
* ``layers``：走过哪些层。层覆盖取「路由层 ∪ 已观测事件的声明层」——
  低风险病例的 L1 只体现为 ``l1_clear`` 事件、危机病例的 L5 只体现为
  ``episode_summary_ready`` 事件，只看路由会漏判；
* ``termination``：终止原因是否与预期一致；病例写 ``null`` 时不允许出现终止；
* ``excluded``：要求被排除的候选诊断是否真的进了 L3 excluded 列表；
* ``guard``：一致性守卫的阻断性违规必须为零（「矛盾输出被记录并终止该次评测」）；
* ``errors``：系统 API 错误码应为空；
* ``min_turns``：轮数下限。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.client.base import NON_FATAL_CODES
from patient_agent.recorder import SessionRecorder
from patient_agent.schemas import PatientScenario

#: 只统计这五层；``orchestrator`` 之类的非层级标签不进覆盖统计。
_OBSERVABLE_LAYERS = ("L1", "L2", "L3", "L4", "L5")

#: 这些错误码是「按设计走到专门路径」的信号（安全转介、人工复核），
#: 出现在记录里属于正常，不算错误。
_DESIGNED_CODES = {code.value for code in NON_FATAL_CODES}


class CheckResult(BaseModel):
    """单项检查的结论。"""

    model_config = ConfigDict(extra="ignore")

    name: str
    ok: bool
    detail: str = ""


class ScenarioEvaluation(BaseModel):
    """一个病例的评测结论。"""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    scenario_version: str = ""
    scenario_title: str = ""
    session_id: str = ""
    ok: bool = True

    checks: list[CheckResult] = Field(default_factory=list)

    observed_events: list[str] = Field(default_factory=list)
    missing_events: list[str] = Field(default_factory=list)
    forbidden_hits: list[str] = Field(default_factory=list)

    layers_observed: list[str] = Field(default_factory=list)
    layers_expected: list[str] = Field(default_factory=list)
    missing_layers: list[str] = Field(default_factory=list)

    termination_expected: str | None = None
    termination_observed: str | None = None

    excluded_expected: list[str] = Field(default_factory=list)
    excluded_observed: list[str] = Field(default_factory=list)
    excluded_missing: list[str] = Field(default_factory=list)

    guard_violations: list[dict] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    # 非设计内的错误码（真故障），``errors`` 检查只看这一项。
    fatal_error_codes: list[str] = Field(default_factory=list)

    turn_count: int = 0
    treatment_turns: int = 0
    min_turns: int = 1
    stop_reason: str = ""
    timing: dict[str, float] = Field(default_factory=dict)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [item for item in self.checks if not item.ok]


def _observed_layers_from_events(scenario: PatientScenario, observed: set[str]) -> set[str]:
    """从「已观测到的预期事件」反推层覆盖。"""

    layers: set[str] = set()
    for event in scenario.expected.events:
        if event.event_id in observed and event.layer in _OBSERVABLE_LAYERS:
            layers.add(event.layer)
    return layers


from .diagnostic_catalog import diagnostic_codes


def _collect_excluded_codes(recorder: SessionRecorder) -> list[str]:
    """从疗程总结的 L3 结果里取被排除的候选诊断码。"""

    summary = recorder.episode_summary or {}
    l3_result = summary.get("l3_result") or {}
    codes: list[str] = []
    for item in l3_result.get("excluded") or []:
        code = item.get("code") if isinstance(item, dict) else str(item)
        if code:
            for canonical in diagnostic_codes(code):
                if canonical not in codes:
                    codes.append(canonical)
    return codes


def _locate_violation(violation: dict) -> str:
    """把守卫违规渲染成「CODE（第N轮）」，方便定位到具体轮次。

    失败详情只报错误码的话，一次 30 轮的会话里仍要人工翻 recorder 才能找到出问题的那一轮；
    turn_index 由 recorder 落盘时就带上，这里只是把它带进报告。
    """

    code = violation.get("code", "?")
    turn_index = violation.get("turn_index")
    if isinstance(turn_index, int):
        return f"{code}（第{turn_index}轮）"
    return code


def evaluate_scenario(
    scenario: PatientScenario,
    recorder: SessionRecorder,
    *,
    blocking_violations_only: bool = True,
) -> ScenarioEvaluation:
    """对照 ``scenario.expected`` 评测一次运行记录。"""

    expected = scenario.expected
    observed_events = recorder.all_events()
    observed_set = set(observed_events)

    checks: list[CheckResult] = []

    # ---------------------------------------------------------- 必需 / 禁止事件
    missing_events = [
        item.event_id
        for item in expected.events
        if item.required and item.event_id not in observed_set
    ]
    checks.append(
        CheckResult(
            name="required_events",
            ok=not missing_events,
            detail="全部命中" if not missing_events else "缺失：" + "、".join(missing_events),
        )
    )

    forbidden_hits = [item for item in expected.forbidden_events if item in observed_set]
    checks.append(
        CheckResult(
            name="forbidden_events",
            ok=not forbidden_hits,
            detail="未出现禁止事件" if not forbidden_hits else "出现了：" + "、".join(forbidden_hits),
        )
    )

    # ------------------------------------------------------------------ 层覆盖
    layers_observed = sorted(
        (set(recorder.layers_visited()) & set(_OBSERVABLE_LAYERS))
        | _observed_layers_from_events(scenario, observed_set)
    )
    missing_layers = [item for item in expected.expected_layers if item not in layers_observed]
    checks.append(
        CheckResult(
            name="layers",
            ok=not missing_layers,
            detail=(
                "覆盖 " + "、".join(layers_observed)
                if not missing_layers
                else "未覆盖：" + "、".join(missing_layers)
            ),
        )
    )

    # -------------------------------------------------------------------- 终止
    termination_observed = recorder.termination_observed()
    if expected.termination_reason is None:
        termination_ok = termination_observed is None
        termination_detail = (
            "按预期未出现疗程终止"
            if termination_ok
            else f"不应终止，却以 {termination_observed} 结束"
        )
    else:
        termination_ok = termination_observed == expected.termination_reason
        termination_detail = (
            f"终止原因一致：{termination_observed}"
            if termination_ok
            else f"终止原因不符：期望 {expected.termination_reason}，实际 {termination_observed}"
        )
    checks.append(CheckResult(name="termination", ok=termination_ok, detail=termination_detail))

    # ------------------------------------------------------------ 候选排除条件
    excluded_observed = _collect_excluded_codes(recorder)
    excluded_missing = [item for item in expected.expected_excluded_conditions if item not in excluded_observed]
    checks.append(
        CheckResult(
            name="excluded_conditions",
            ok=not excluded_missing,
            detail=(
                "已排除 " + "、".join(excluded_observed)
                if excluded_observed
                else "无被排除的候选"
            )
            if not excluded_missing
            else "未排除：" + "、".join(excluded_missing),
        )
    )

    # ---------------------------------------------------------------- 一致性守卫
    guard_violations = recorder.guard_violations()
    if blocking_violations_only:
        guard_failures = [
            item for item in guard_violations if item.get("blocking", True)
        ]
    else:
        guard_failures = guard_violations
    checks.append(
        CheckResult(
            name="guard",
            ok=not guard_failures,
            detail=(
                "无阻断性违规"
                if not guard_failures
                else f"{len(guard_failures)} 条阻断性违规："
                + "、".join(sorted({_locate_violation(item) for item in guard_failures}))
            ),
        )
    )

    # ------------------------------------------------------------------ 错误码
    error_codes = recorder.error_codes()
    fatal_error_codes = [code for code in error_codes if code not in _DESIGNED_CODES]
    checks.append(
        CheckResult(
            name="errors",
            ok=not fatal_error_codes,
            detail=(
                "无错误码"
                if not fatal_error_codes
                else "出现：" + "、".join(fatal_error_codes)
            ),
        )
    )

    # ---------------------------------------------------------------- 轮数下限
    checks.append(
        CheckResult(
            name="min_turns",
            ok=recorder.turn_count >= expected.min_turns,
            detail=f"{recorder.turn_count} / 下限 {expected.min_turns}",
        )
    )

    return ScenarioEvaluation(
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.version,
        scenario_title=scenario.title,
        session_id=recorder.session_id,
        ok=all(item.ok for item in checks),
        checks=checks,
        observed_events=observed_events,
        missing_events=missing_events,
        forbidden_hits=forbidden_hits,
        layers_observed=layers_observed,
        layers_expected=list(expected.expected_layers),
        missing_layers=missing_layers,
        termination_expected=expected.termination_reason,
        termination_observed=termination_observed,
        excluded_expected=list(expected.expected_excluded_conditions),
        excluded_observed=excluded_observed,
        excluded_missing=excluded_missing,
        guard_violations=guard_violations,
        error_codes=error_codes,
        fatal_error_codes=fatal_error_codes,
        turn_count=recorder.turn_count,
        treatment_turns=recorder.treatment_turns,
        min_turns=expected.min_turns,
        stop_reason=recorder.stop_reason,
        timing=recorder.timing_summary(),
    )
