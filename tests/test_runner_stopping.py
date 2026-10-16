"""PA5/PA6：场景运行器与停止条件。

分两层验：

* ``evaluate_stop`` 是纯函数，直接造记录与响应验判序——「系统终止」永远最硬，
  其次是安全暂停/人工复核，再次是患者侧违规与目标事件，``max_turns`` 垫底；
* 运行器跑真会话，验证停止条件真的让循环停了：危机一轮就收、目标事件覆盖即停、
  ``max_turns`` 兜底、暂停按计划插入、每轮耗时被记下、API 异常被路由到 L5 而非 L1。
"""

from __future__ import annotations

from patient_agent.client import MockConfig, MockSystemApiClient
from patient_agent.client.base import TurnResponse
from patient_agent.config import Settings
from patient_agent.errors import ErrorCode, SystemApiError
from patient_agent.factory import build_stack
from patient_agent.generator import PatientAction
from patient_agent.guard import GuardCode, Violation
from patient_agent.llm import RuleBasedLLM
from patient_agent.recorder import SessionRecorder
from patient_agent.runner import RunOptions
from patient_agent.schemas import StopPolicy
from patient_agent.stopping import evaluate_stop


# ------------------------------------------------------------------ 构造小工具
def reply(**kwargs) -> TurnResponse:
    return TurnResponse(session_id="S-1", patient_reply="嗯。", **kwargs)


def recorder_with(
    *, violations: list[Violation] | None = None, events: list[str] | None = None
) -> SessionRecorder:
    record = SessionRecorder(scenario_id="S-test")
    record.add_turn(
        turn_index=1,
        action=PatientAction(kind="say", text="嗯", reason="单测"),
        patient_text="嗯",
        response=TurnResponse(session_id="S-1", patient_reply="嗯。", events=list(events or [])),
        violations=violations,
    )
    return record


def decide(
    policy: StopPolicy,
    *,
    turn_count: int = 1,
    response: TurnResponse | None = None,
    recorder: SessionRecorder | None = None,
):
    return evaluate_stop(
        policy,
        turn_count=turn_count,
        response=response or reply(),
        recorder=recorder or recorder_with(),
    )


# ============================================================ evaluate_stop 判序
def test_system_termination_beats_everything() -> None:
    decision = decide(
        StopPolicy(max_turns=3, target_events=["never_happens"]),
        turn_count=1,
        response=reply(session_ended=True, termination_reason="dropout"),
    )
    assert decision.should_stop is True
    assert decision.system_terminated is True
    assert "dropout" in decision.reason


def test_session_ended_without_reason_still_stops() -> None:
    decision = decide(
        StopPolicy(max_turns=99),
        response=reply(session_ended=True),
    )
    assert decision.should_stop is True
    assert decision.system_terminated is True
    assert decision.reason == "系统终止：session_ended"


def test_safety_hold_stops_and_is_system_level() -> None:
    decision = decide(
        StopPolicy(stop_on_crisis=True, max_turns=99),
        response=reply(error_code=ErrorCode.SAFETY_HOLD),
    )
    assert decision.should_stop is True
    assert decision.system_terminated is True


def test_safety_hold_can_be_declined() -> None:
    decision = decide(
        StopPolicy(stop_on_crisis=False, max_turns=99),
        response=reply(error_code=ErrorCode.SAFETY_HOLD),
    )
    assert decision.should_stop is False


def test_human_review_stops_and_can_be_declined() -> None:
    eager = decide(
        StopPolicy(stop_on_human_review=True, max_turns=99),
        response=reply(error_code=ErrorCode.HUMAN_REVIEW),
    )
    assert eager.should_stop is True and eager.system_terminated is True

    patient = decide(
        StopPolicy(stop_on_human_review=False, max_turns=99),
        response=reply(error_code=ErrorCode.HUMAN_REVIEW),
    )
    assert patient.should_stop is False


def test_blocking_violation_is_a_test_stop_not_system_stop() -> None:
    record = recorder_with(
        violations=[Violation(code=GuardCode.NEVER_DISCLOSE, message="说了不该说的")]
    )
    decision = decide(StopPolicy(stop_on_violation=True, max_turns=99), recorder=record)
    assert decision.should_stop is True
    assert decision.system_terminated is False, "这是患者侧的问题，不是系统终止"


def test_non_blocking_violation_is_only_recorded() -> None:
    record = recorder_with(
        violations=[
            Violation(code=GuardCode.FACT_MISMATCH, message="轻微漂移", blocking=False)
        ]
    )
    decision = decide(StopPolicy(stop_on_violation=True, max_turns=99), recorder=record)
    assert decision.should_stop is False


def test_target_events_stop_when_fully_covered() -> None:
    record = recorder_with(events=["l3_candidates_generated"])
    hit = decide(
        StopPolicy(target_events=["l3_candidates_generated"], max_turns=99), recorder=record
    )
    assert hit.should_stop is True
    assert "目标事件" in hit.reason
    assert hit.system_terminated is False

    miss = decide(
        StopPolicy(target_events=["l3_candidates_generated", "absent"], max_turns=99),
        recorder=record,
    )
    assert miss.should_stop is False


def test_max_turns_is_the_backstop() -> None:
    decision = decide(StopPolicy(max_turns=5), turn_count=5)
    assert decision.should_stop is True
    assert "max_turns=5" in decision.reason
    assert decision.system_terminated is False


def test_nothing_triggers_means_keep_going() -> None:
    assert decide(StopPolicy(max_turns=5), turn_count=2).should_stop is False


# ============================================================== 运行器端到端行为
def test_crisis_scenario_stops_after_opening_only(runner_factory) -> None:
    record = runner_factory("S03_explicit_crisis").runner.run()

    assert record.turn_count == 1, "危机在开场那一轮就必须停"
    assert record.turns[0].turn_index == 0
    assert record.turns[0].main_layer == "L1"
    assert record.termination_observed() == "crisis_referral"
    assert record.error_codes() == ["SAFETY_HOLD"]
    assert record.stop_reason.startswith("系统终止")
    # 后续层一个都不该碰。
    assert record.layers_visited() == ["L1"]


def test_target_event_stops_s02_without_termination(runner_factory) -> None:
    record = runner_factory("S02_insufficient_info").runner.run()

    assert "l3_candidates_generated" in record.all_events()
    assert "目标事件" in record.stop_reason
    assert record.termination_observed() is None, "S02 只验证回环，不该出现疗程终止"


def test_max_turns_backstop(library) -> None:
    scenario = library.load("S01_low_risk_depression").model_copy(deep=True)
    # 把上限压到 4，逼运行器走到兜底那条分支。
    scenario.expected.min_turns = 1
    scenario.stop = StopPolicy(max_turns=4)
    client = MockSystemApiClient(MockConfig())
    stack = build_stack(
        scenario,
        Settings(llm_mode="offline", system_api_mode="mock"),
        client=client,
        llm=RuleBasedLLM(),
        options=RunOptions(),
    )
    record = stack.runner.run()

    assert record.stop_reason == "达到 max_turns=4"
    assert record.turn_count <= 5
    assert record.termination_observed() is None


def test_opening_is_turn_zero(runner_factory) -> None:
    record = runner_factory("S01_low_risk_depression").runner.run()
    first = record.turns[0]
    assert first.turn_index == 0
    assert first.action_kind == "say"
    assert first.patient_text, "开场必须说点什么"


def test_pause_is_inserted_where_planned(runner_factory) -> None:
    record = runner_factory("S05_comorbid_anxiety_depression").runner.run()
    pauses = [turn for turn in record.turns if turn.action_kind == "set_pause"]
    assert pauses, "S05 规划了一次主动暂停"
    assert pauses[0].turn_index == 8, "pause_after_turn=8，应当插在第 8 轮"
    assert any("暂停" in note for note in record.notes + [pauses[0].note])


def test_elapsed_ms_is_recorded(runner_factory) -> None:
    record = runner_factory("S01_low_risk_depression").runner.run()
    assert record.turns[0].elapsed_ms > 0
    timing = record.timing_summary()
    assert timing["count"] == record.turn_count
    assert timing["total_ms"] > 0
    assert timing["max_ms"] >= timing["avg_ms"]


def test_full_run_collects_artifacts(runner_factory) -> None:
    record = runner_factory("S01_low_risk_depression").runner.run()
    assert record.final_state, "收尾要拉开发状态"
    assert record.episode_summary is not None
    assert record.history


# ------------------------------------------------------------- 系统 API 异常的收敛
class _BoomClient(MockSystemApiClient):
    """第 2 次 send_message 就炸，用来验异常轮的记录与路由。"""

    name = "boom"

    def __init__(self) -> None:
        super().__init__(MockConfig())
        self.calls = 0

    def send_message(self, session_id: str, text: str) -> TurnResponse:
        self.calls += 1
        if self.calls >= 2:
            raise SystemApiError("编排器连接中断", error_code=ErrorCode.API_ERROR)
        return super().send_message(session_id, text)


def _run_with_boom(library, *, tolerate: bool) -> SessionRecorder:
    scenario = library.load("S01_low_risk_depression")
    stack = build_stack(
        scenario,
        Settings(llm_mode="offline", system_api_mode="mock"),
        client=_BoomClient(),
        llm=RuleBasedLLM(),
        options=RunOptions(tolerate_errors=tolerate),
    )
    return stack.runner.run()


def test_api_error_is_swallowed_and_routed_to_l5(library) -> None:
    record = _run_with_boom(library, tolerate=True)

    error_turns = [turn for turn in record.turns if turn.error_code]
    assert error_turns, "异常轮应当被记录下来，否则报告统计不到错误类型"
    assert error_turns[0].main_layer == "L5", "异常轮没有真实层参与，不能顶替 L1"
    assert error_turns[0].route_intent == "error"
    assert record.stop_reason == "系统 API 异常：API_ERROR"
    assert record.error_codes() == ["API_ERROR"]


def test_api_error_reraises_when_intolerance_asked(library) -> None:
    import pytest

    with pytest.raises(SystemApiError):
        _run_with_boom(library, tolerate=False)


def test_error_turn_helper_does_not_claim_l1(library) -> None:
    """直接盯住 ``_error_turn``：``RouteInfo`` 默认是 L1，异常轮必须改掉。"""

    scenario = library.load("S01_low_risk_depression")
    stack = build_stack(
        scenario,
        Settings(llm_mode="offline", system_api_mode="mock"),
        client=MockSystemApiClient(MockConfig()),
        llm=RuleBasedLLM(),
        options=RunOptions(),
    )
    turn = stack.runner._error_turn(  # noqa: SLF001 - 单测要看这条路
        "S-x", SystemApiError("炸了", error_code=ErrorCode.TIMEOUT)
    )
    assert turn.route.main_layer == "L5"
    assert turn.route.intent == "error"
    assert turn.route.reason == "炸了"
    assert turn.error_code == ErrorCode.TIMEOUT
    assert turn.patient_reply == ""
