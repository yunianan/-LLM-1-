"""端到端验收：把任务说明里那五条验收场景，逐条钉成断言。

这里的用例都跑真的整条链（生成器 → 守卫 → 运行器 → Mock 五层桩 → 评测器），
用的还是离线确定性模式，所以结果可复现。
"""

from __future__ import annotations

from patient_agent.client.base import NON_FATAL_CODES
from patient_agent.errors import ErrorCode
from patient_agent.factory import build_library, run_all, run_and_evaluate

ALL_IDS = [
    "S01_low_risk_depression",
    "S02_insufficient_info",
    "S03_explicit_crisis",
    "S04_bipolar_exclusion",
    "S05_comorbid_anxiety_depression",
    "S06_treatment_dropout",
]


def first_turn(record, event: str) -> int | None:
    for turn in record.turns:
        if event in turn.events:
            return turn.turn_index
    return None


# ------------------------------------------------------------------ 整库回放
def test_batch_replays_all_six_and_all_pass(settings) -> None:
    outcome = run_all(settings)

    assert [item.scenario_id for item in outcome.records] == ALL_IDS
    assert outcome.report.total == 6
    assert outcome.report.passed == 6
    assert outcome.report.failed == 0
    assert outcome.report.pass_rate == 1.0
    for evaluation in outcome.evaluations:
        failures = "、".join(
            f"{check.name}（{check.detail}）" for check in evaluation.failed_checks
        )
        assert evaluation.ok, f"{evaluation.scenario_id} 未通过：{failures}"


def test_batch_report_covers_every_layer(settings) -> None:
    outcome = run_all(settings)
    coverage = outcome.report.layer_coverage
    # L1 只体现为 l1_clear（非危机）或安全路径（危机），L5 只在结案时出现；
    # 两处都要靠「事件声明的层」补齐，不能只看路由。
    assert coverage.get("L1", 0) == 6
    assert coverage.get("L5", 0) >= 4
    assert outcome.report.fatal_error_counts == {}, "不许出现非设计内故障"
    assert outcome.report.fact_consistency_violations == 0, "不许出现与病例相矛盾的事实"


def test_termination_distribution_matches_case_book(settings) -> None:
    outcome = run_all(settings)
    assert outcome.report.termination_counts == {
        "未终止": 1,
        "completed_improved": 2,
        "completed_no_change": 1,
        "crisis_referral": 1,
        "dropout": 1,
    }


# ------------------------------------------------- 验收场景 1：画像确认 → L3
def test_criterion_1_profile_confirmation_then_l3(settings) -> None:
    scenario = build_library(settings).load("S01_low_risk_depression")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    observed = set(record.all_events())
    assert "l2_profile_confirmed" in observed, "L2 要走完画像确认"
    assert "l3_candidates_generated" in observed, "确认之后应进入 L3 并产出候选"
    assert "L3" in record.layers_visited()
    confirmed_at = first_turn(record, "l2_profile_confirmed")
    candidates_at = first_turn(record, "l3_candidates_generated")
    assert confirmed_at is not None and candidates_at is not None
    # 信息足够时 L3 会在确认的同一轮直接出候选，所以是「不早于」而不是「严格晚于」。
    assert confirmed_at <= candidates_at, "L3 出候选不能早于画像确认"
    assert record.termination_observed() == "completed_improved"


# ------------------------------------------- 验收场景 2：L3 信息不足回退 L2 补字段
def test_criterion_2_insufficient_info_loops_back_and_supplements(settings) -> None:
    scenario = build_library(settings).load("S02_insufficient_info")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    observed = set(record.all_events())
    assert "l3_insufficient_return_l2" in observed
    assert "l2_supplement_requested" in observed

    # 挂在回环事件上的「被锁事实」，只有回环发生后才能说出口。
    gated = [
        item
        for item in scenario.revealable_facts()
        if "l3_insufficient_return_l2" in item.unlock_after_events
    ]
    assert gated, "S02 应当有被回环事件解锁的补充事实"
    expected_utterances = [item.utterance for item in gated if item.utterance]

    loop_at = first_turn(record, "l3_insufficient_return_l2")
    assert loop_at is not None
    spoken_after = " ".join(
        turn.patient_text for turn in record.turns if turn.turn_index > loop_at
    )
    assert any(text in spoken_after for text in expected_utterances), (
        "回环之后必须补出被锁的那条症状"
    )

    # 回环是为了补信息，不该以终止收场。
    assert record.termination_observed() is None
    assert evaluation.stop_reason.startswith("目标事件")


# --------------------------------------------- 验收场景 3：危机只走 L1 安全路径
def test_criterion_3_crisis_touches_l1_only(settings) -> None:
    scenario = build_library(settings).load("S03_explicit_crisis")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    assert record.turn_count == 1, "危机在开场即被识别，不应继续对话"
    assert record.layers_visited() == ["L1"]
    assert record.termination_observed() == "crisis_referral"
    assert record.turns[0].waiting_type == "safety_hold"
    assert record.error_codes() == ["SAFETY_HOLD"]

    downstream = {
        "l2_scale_started",
        "l2_scale_completed",
        "l2_profile_ready",
        "l2_profile_confirmed",
        "l3_candidates_generated",
        "l4_treatment_started",
        "l4_improvement_detected",
        "episode_summary_ready",
    }
    assert not (set(record.all_events()) & downstream), "后续层一个都不能被调用"
    assert record.turns[0].system_reply, "安全路径也要给患者一句可读回应"


# -------------------------------------- 验收场景 4：双相被排除，患者不迎合系统
def test_criterion_4_bipolar_excluded_while_patient_stays_faithful(settings) -> None:
    scenario = build_library(settings).load("S04_bipolar_exclusion")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    assert "D008" in evaluation.excluded_observed and "D009" in evaluation.excluded_observed
    assert evaluation.excluded_missing == []

    layers = record.final_state["layers"]["L3"]
    candidate_codes = [item["code"] for item in layers["candidates"]]
    excluded_codes = [item["code"] for item in layers["excluded"]]
    assert "D013" in candidate_codes, "抑郁本身不该被牵连排除"
    assert "D008" not in candidate_codes and "D009" not in candidate_codes
    assert "D008" in excluded_codes and "D009" in excluded_codes

    # 不迎合：患者在被暗示「也许是双相」时仍按病例否认躁狂史。
    assert any(
        turn.patient_fact_id == "exclusion_absent:manic_history" and turn.patient_denial
        for turn in record.turns
    ), "必须按病例否认躁狂史，而不是顺着系统说"
    assert record.termination_observed() == "completed_improved"


# ------------------------------------------- 验收场景 5：改善 / 无变化 / 退出三条轨迹
def test_criterion_5_three_trajectories(settings) -> None:
    library = build_library(settings)
    expected = {
        "S01_low_risk_depression": "completed_improved",
        "S05_comorbid_anxiety_depression": "completed_no_change",
        "S06_treatment_dropout": "dropout",
    }
    for scenario_id, reason in expected.items():
        record, evaluation = run_and_evaluate(library.load(scenario_id), settings)
        assert evaluation.ok, scenario_id
        assert record.termination_observed() == reason, scenario_id
        assert record.turns[-1].session_ended is True, scenario_id
        assert record.episode_summary is not None, "结案后应能读到疗程总结"
        assert record.episode_summary["termination_reason"] == reason


def test_dropout_leaves_summary_and_history(settings) -> None:
    scenario = build_library(settings).load("S06_treatment_dropout")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    assert "l4_dropout_detected" in record.all_events()
    assert record.episode_summary["termination_reason"] == "dropout"
    assert any(item["session_id"] == record.session_id for item in record.history)
    assert record.history[0]["termination_reason"] == "dropout"
    assert record.episode_summary["l3_result"], "总结里要带上 L3 的结论"


# ------------------------------------------------------------ 暂停 / 恢复链路
def test_pause_resume_chain_in_s05(settings) -> None:
    scenario = build_library(settings).load("S05_comorbid_anxiety_depression")
    record, evaluation = run_and_evaluate(scenario, settings)

    assert evaluation.ok
    pause_turns = [turn for turn in record.turns if turn.action_kind == "set_pause"]
    assert [turn.patient_text for turn in pause_turns] == ["[暂停]", "[恢复]"]
    assert pause_turns[0].waiting_type == "pause"
    assert pause_turns[1].waiting_type != "pause", "恢复之后不该还停在暂停态"
    # 暂停不该影响疗程走完。
    assert record.termination_observed() == "completed_no_change"


def test_episode_ledger_books_every_turn_exactly_once(settings) -> None:
    """疗程总结的逐轮账本里，每个真正被系统处理的轮次只出现一次。

    真正会出错的只有「一次调用被记两笔」——暂停/恢复、量表答完这类「合并措辞」
    的路径最容易犯。所以拿「记账条数」和「实际处理的轮次数」对一下，多记的就
    藏不住了。（轮号本身会重复，不能拿轮号唯一性来判。）
    """

    library = build_library(settings)
    for scenario_id in ALL_IDS:
        record, evaluation = run_and_evaluate(library.load(scenario_id), settings)
        summary = record.episode_summary
        if record.termination_observed() is None:
            assert summary is None, f"{scenario_id} 未终止时不该有总结"
            continue

        assert summary is not None, scenario_id
        reviews = summary["l5_reviews"]
        booked = [
            turn
            for turn in record.turns
            if turn.error_code is None or ErrorCode(turn.error_code) in NON_FATAL_CODES
        ]
        assert reviews, f"{scenario_id} 结案了却没有逐轮复盘"
        assert len(reviews) == len(booked), (
            f"{scenario_id} 记账 {len(reviews)} 条，实际只处理了 {len(booked)} 轮"
        )
        assert sum(summary["route_summary"].values()) == len(reviews), scenario_id


def test_every_scenario_matches_its_declared_expectations(settings) -> None:
    """把病例自己写的 expected 与实测对一遍，防止有人改了病例却没改断言。"""

    library = build_library(settings)
    for scenario_id in ALL_IDS:
        scenario = library.load(scenario_id)
        record, evaluation = run_and_evaluate(scenario, settings)

        assert evaluation.termination_observed == scenario.expected.termination_reason
        assert evaluation.missing_events == [], scenario_id
        assert evaluation.forbidden_hits == [], scenario_id
        assert evaluation.missing_layers == [], scenario_id
        assert evaluation.excluded_missing == [], scenario_id
        assert record.turn_count >= scenario.expected.min_turns
