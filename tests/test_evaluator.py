"""PA7：评测器。

评测器只读记录、不跑会话，所以这里大多用手工造的「记录 + 病例」组合精确校验每一条
判分口径；最后再补一条真跑 S01 的集成用例，证明整套口径拼起来是通的。
"""

from __future__ import annotations

from patient_agent.client.base import RouteInfo, TurnResponse
from patient_agent.errors import ErrorCode
from patient_agent.evaluator import evaluate_scenario
from patient_agent.factory import run_and_evaluate
from patient_agent.generator import PatientAction
from patient_agent.guard import GuardCode, Violation
from patient_agent.recorder import SessionRecorder
from patient_agent.schemas import ExpectedEvent, ExpectedOutcome


# ------------------------------------------------------------------ 造样例
def scenario_with(library, **expected) -> object:
    """拿 S03 改 ``expected``：只评测不运行，底子用哪条无所谓。"""

    scenario = library.load("S03_explicit_crisis").model_copy(deep=True)
    scenario.expected = ExpectedOutcome(**expected)
    return scenario


def empty_record(scenario) -> SessionRecorder:
    return SessionRecorder(scenario_id=scenario.scenario_id)


def add(
    record: SessionRecorder,
    *,
    index: int = 1,
    events: tuple[str, ...] = (),
    layer: str = "L1",
    secondary: tuple[str, ...] = (),
    intent: str = "",
    error_code: ErrorCode | None = None,
    violation: Violation | None = None,
    termination: str | None = None,
    session_ended: bool = False,
) -> None:
    record.add_turn(
        turn_index=index,
        action=PatientAction(kind="say", text="嗯", reason="单测"),
        patient_text="嗯",
        response=TurnResponse(
            session_id="S-1",
            patient_reply="系统回复",
            events=list(events),
            route=RouteInfo(
                main_layer=layer, secondary_layers=list(secondary), intent=intent
            ),
            error_code=error_code,
            session_ended=session_ended,
            termination_reason=termination,
        ),
        violations=[violation] if violation else None,
    )


def check(evaluation, name: str):
    return next(item for item in evaluation.checks if item.name == name)


# ------------------------------------------------------------------ 事件
def test_missing_required_event_fails(library) -> None:
    scenario = scenario_with(
        library,
        events=[ExpectedEvent(event_id="need_me", layer="L2")],
        expected_layers=["L2"],
    )
    record = empty_record(scenario)
    add(record, events=("something_else",), layer="L2")

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.ok is False
    assert evaluation.missing_events == ["need_me"]
    assert check(evaluation, "required_events").ok is False


def test_optional_event_may_be_absent(library) -> None:
    scenario = scenario_with(
        library,
        events=[ExpectedEvent(event_id="nice_to_have", layer="L2", required=False)],
        expected_layers=[],
    )
    evaluation = evaluate_scenario(scenario, empty_record(scenario))
    assert evaluation.missing_events == []
    assert check(evaluation, "required_events").ok is True


def test_forbidden_event_fails(library) -> None:
    scenario = scenario_with(library, forbidden_events=["never_ever"], expected_layers=[])
    record = empty_record(scenario)
    add(record, events=("never_ever",), layer="L2")

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.ok is False
    assert evaluation.forbidden_hits == ["never_ever"]
    assert check(evaluation, "forbidden_events").ok is False


# ------------------------------------------------------------------ 层覆盖
def test_layers_union_route_and_declared_event_layer(library) -> None:
    """低风险病例的 L1 只体现为 ``l1_clear``；只看路由会漏判。"""

    scenario = scenario_with(
        library,
        events=[ExpectedEvent(event_id="l1_clear", layer="L1")],
        expected_layers=["L1", "L2"],
    )
    record = empty_record(scenario)
    # 路由是 L2，但观测到了声明为 L1 的事件。
    add(record, events=("l1_clear",), layer="L2")

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.layers_observed == ["L1", "L2"]
    assert evaluation.missing_layers == []
    assert check(evaluation, "layers").ok is True


def test_missing_layer_fails(library) -> None:
    scenario = scenario_with(library, expected_layers=["L1", "L5"])
    record = empty_record(scenario)
    add(record, layer="L2")

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.missing_layers == ["L1", "L5"]
    assert check(evaluation, "layers").ok is False


def test_secondary_layer_counts_as_visited(library) -> None:
    scenario = scenario_with(library, expected_layers=["L3", "L4"])
    record = empty_record(scenario)
    add(record, layer="L4", secondary=("L3",))

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.layers_observed == ["L3", "L4"]


# -------------------------------------------------------------------- 终止
def test_expected_none_means_no_termination_allowed(library) -> None:
    scenario = scenario_with(library, termination_reason=None, expected_layers=[])
    record = empty_record(scenario)
    add(record, termination="dropout", session_ended=True)

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.termination_observed == "dropout"
    assert check(evaluation, "termination").ok is False


def test_expected_none_and_truly_unterminated_passes(library) -> None:
    scenario = scenario_with(library, termination_reason=None, expected_layers=[])
    record = empty_record(scenario)
    add(record)

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.termination_observed is None
    assert check(evaluation, "termination").ok is True


def test_termination_mismatch_fails_and_match_passes(library) -> None:
    scenario = scenario_with(library, termination_reason="dropout", expected_layers=[])

    matched = empty_record(scenario)
    add(matched, termination="dropout", session_ended=True)
    assert check(evaluate_scenario(scenario, matched), "termination").ok is True

    mismatched = empty_record(scenario)
    add(mismatched, termination="completed_improved", session_ended=True)
    bad = evaluate_scenario(scenario, mismatched)
    assert check(bad, "termination").ok is False
    assert "期望 dropout" in check(bad, "termination").detail


# ---------------------------------------------------------------- 候选排除
def test_excluded_conditions_read_from_episode_summary(library) -> None:
    scenario = scenario_with(
        library, expected_excluded_conditions=["bipolar"], expected_layers=[]
    )
    record = empty_record(scenario)
    add(record)
    record.episode_summary = {
        "l3_result": {"excluded": [{"code": "bipolar"}, {"code": "ptsd"}]}
    }

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.excluded_observed == ["D008", "D009", "D018"]
    assert evaluation.excluded_missing == []
    assert check(evaluation, "excluded_conditions").ok is True


def test_missing_exclusion_fails(library) -> None:
    scenario = scenario_with(
        library, expected_excluded_conditions=["bipolar"], expected_layers=[]
    )
    record = empty_record(scenario)
    add(record)
    record.episode_summary = {"l3_result": {"excluded": [{"code": "ptsd"}]}}

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.excluded_missing == ["D008", "D009"]
    assert check(evaluation, "excluded_conditions").ok is False


def test_excluded_tolerates_plain_strings(library) -> None:
    scenario = scenario_with(
        library, expected_excluded_conditions=["bipolar"], expected_layers=[]
    )
    record = empty_record(scenario)
    add(record)
    record.episode_summary = {"l3_result": {"excluded": ["bipolar"]}}

    assert evaluate_scenario(scenario, record).excluded_observed == ["D008", "D009"]


# ---------------------------------------------------------------- 守卫 / 错误
def test_guard_counts_only_blocking_by_default(library) -> None:
    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(
        record,
        violation=Violation(code=GuardCode.SEVERITY_INFLATION, message="夸张", blocking=False),
    )

    evaluation = evaluate_scenario(scenario, record)
    assert check(evaluation, "guard").ok is True
    assert evaluation.guard_violations, "非阻断违规仍要留痕"


def test_guard_lenient_mode_counts_everything(library) -> None:
    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(
        record,
        violation=Violation(code=GuardCode.SEVERITY_INFLATION, message="夸张", blocking=False),
    )

    evaluation = evaluate_scenario(scenario, record, blocking_violations_only=False)
    assert check(evaluation, "guard").ok is False


def test_blocking_violation_fails_even_in_default_mode(library) -> None:
    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(record, violation=Violation(code=GuardCode.NEVER_DISCLOSE, message="越界"))

    evaluation = evaluate_scenario(scenario, record)
    assert check(evaluation, "guard").ok is False
    assert "NEVER_DISCLOSE" in check(evaluation, "guard").detail


def test_guard_detail_locates_the_offending_turn(library) -> None:
    """验收要求「失败可定位到具体轮次」，所以详情里必须带轮号。"""

    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(record, index=1)
    add(record, index=2, violation=Violation(code=GuardCode.NEVER_DISCLOSE, message="越界"))
    add(record, index=3)

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.guard_violations[0]["turn_index"] == 2
    assert "NEVER_DISCLOSE（第2轮）" in check(evaluation, "guard").detail


def test_guard_detail_dedupes_same_code_on_one_turn(library) -> None:
    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(record, index=1, violation=Violation(code=GuardCode.NEVER_DISCLOSE, message="越界"))

    evaluation = evaluate_scenario(scenario, record)
    assert check(evaluation, "guard").detail.count("NEVER_DISCLOSE") == 1


def test_designed_error_code_is_not_a_failure(library) -> None:
    """SAFETY_HOLD 是安全路径信号，不该被当成故障。"""

    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(record, error_code=ErrorCode.SAFETY_HOLD)

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.error_codes == ["SAFETY_HOLD"]
    assert evaluation.fatal_error_codes == []
    assert check(evaluation, "errors").ok is True


def test_fatal_error_code_fails(library) -> None:
    scenario = scenario_with(library, expected_layers=[])
    record = empty_record(scenario)
    add(record, error_code=ErrorCode.API_ERROR)

    evaluation = evaluate_scenario(scenario, record)
    assert evaluation.fatal_error_codes == ["API_ERROR"]
    assert check(evaluation, "errors").ok is False


# ------------------------------------------------------------------ 轮数下限
def test_min_turns_check(library) -> None:
    scenario = scenario_with(library, expected_layers=[], min_turns=3)
    record = empty_record(scenario)
    add(record, index=1)

    evaluation = evaluate_scenario(scenario, record)
    assert check(evaluation, "min_turns").ok is False
    assert "1 / 下限 3" in check(evaluation, "min_turns").detail


def test_failed_checks_lists_only_failures(library) -> None:
    scenario = scenario_with(
        library,
        events=[ExpectedEvent(event_id="need_me", layer="L2")],
        expected_layers=["L5"],
    )
    record = empty_record(scenario)
    add(record)  # 至少一轮，避免 min_turns 也一起挂
    evaluation = evaluate_scenario(scenario, record)
    names = {item.name for item in evaluation.failed_checks}
    assert names == {"required_events", "layers"}
    assert evaluation.ok is False


def test_evaluation_carries_timing_and_stop_reason(library, settings) -> None:
    scenario = library.load("S01_low_risk_depression")
    record, evaluation = run_and_evaluate(scenario, settings)
    assert evaluation.ok is True
    assert evaluation.timing["count"] == record.turn_count
    assert evaluation.timing["total_ms"] > 0
    assert evaluation.stop_reason
