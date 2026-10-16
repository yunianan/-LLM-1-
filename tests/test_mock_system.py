"""完整系统 API 契约 + 内置 Mock 五层桩。

这一层要证明两件事：
1. 八个接口的签名与返回结构与任务说明第五节一致（上层可以整体替换实现）；
2. 桩自己按「L1 风险 → L2 访谈/量表/画像 → L3 候选 → L4 治疗 → L5 复盘」走，
   并且把「错误」和「流程分支」区分开——错的是错误码，对的是事件。
"""

from __future__ import annotations

import inspect

import pytest

from patient_agent.client import (
    MockConfig,
    MockSystemApiClient,
    ProfileDecision,
    SystemApiClient,
)
from patient_agent.errors import ErrorCode

#: 任务说明第五节列出的八个接口。
SYSTEM_API = [
    "create_session",
    "send_message",
    "confirm_profile",
    "submit_scale",
    "set_pause",
    "get_dev_state",
    "get_episode_summary",
    "get_session_history",
]


def new_client(**config) -> MockSystemApiClient:
    return MockSystemApiClient(MockConfig(**config))


def answer_scales(client: MockSystemApiClient, session_id: str, response, *, score: int = 1):
    """量表逐题作答，直到不再是「等答题」。"""

    while response.waiting_type == "scale_item" and response.scale_item is not None:
        item = response.scale_item
        response = client.submit_scale(
            session_id, item.item_number, min(score, item.max_score)
        )
    return response


def drive(
    client: MockSystemApiClient,
    session_id: str,
    texts: list[str],
    *,
    confirm: str = "continue",
):
    """发一批患者消息；途中自动答量表、自动做画像确认，返回最后一轮响应。"""

    last = None
    for text in texts:
        last = answer_scales(client, session_id, client.send_message(session_id, text))
        if last.waiting_type == "profile_confirmation":
            last = answer_scales(
                client, session_id, client.confirm_profile(session_id, confirm)
            )
    return last


# ------------------------------------------------------------------ 接口契约
def test_all_eight_interfaces_exist() -> None:
    for name in SYSTEM_API:
        assert callable(getattr(MockSystemApiClient, name, None)), f"缺少接口 {name}"


def test_mock_implements_abstract_client() -> None:
    assert issubclass(MockSystemApiClient, SystemApiClient)
    assert inspect.isabstract(SystemApiClient), "抽象基类应当无法直接实例化"


def test_signatures_match_the_contract() -> None:
    """上层按抽象签名调用，桩的签名不能漂。"""

    for name in SYSTEM_API:
        base = inspect.signature(getattr(SystemApiClient, name))
        impl = inspect.signature(getattr(MockSystemApiClient, name))
        assert list(base.parameters) == list(impl.parameters), name


def test_create_session_returns_snapshot() -> None:
    client = new_client()
    snapshot = client.create_session(locale="zh-CN")
    assert snapshot.session_id
    assert snapshot.locale == "zh-CN"
    assert snapshot.route.main_layer in ("L1", "L2", "L3", "L4", "L5")


def test_unknown_session_raises() -> None:
    client = new_client()
    with pytest.raises(Exception):
        client.get_dev_state("不存在")


# --------------------------------------------------------------------- L1
def test_explicit_crisis_enters_safety_path_only() -> None:
    client = new_client()
    sid = client.create_session().session_id
    response = client.send_message(sid, "我实在撑不下去了，总想着结束这一切，想死。")

    assert response.route.main_layer == "L1"
    assert response.waiting_type == "safety_hold"
    assert response.error_code == ErrorCode.SAFETY_HOLD
    assert response.session_ended is True
    assert response.termination_reason == "crisis_referral"
    assert "l1_safety_activated" in response.events
    assert "crisis_referral" in response.events
    # 危机时不收集更多信息：不该出现任何量表事件。
    assert "l2_scale_started" not in response.events
    assert "l3_candidates_generated" not in response.events


def test_crisis_reply_is_patient_facing() -> None:
    client = new_client()
    sid = client.create_session().session_id
    response = client.send_message(sid, "我想死。")
    assert response.patient_reply, "安全路径也必须给患者一句可读的回应"
    assert "安全" in response.patient_reply


def test_ambiguous_risk_is_noted_but_continues() -> None:
    client = new_client()
    sid = client.create_session().session_id
    response = client.send_message(sid, "有时候觉得活着没意思，撑不太下去。")
    assert "l1_ambiguous_risk_noted" in response.events
    assert response.session_ended is False
    assert response.error_code is None
    assert response.waiting_type != "safety_hold"


def test_clear_risk_emits_l1_clear() -> None:
    client = new_client()
    sid = client.create_session().session_id
    response = client.send_message(sid, "我最近情绪不太好。")
    assert "l1_clear" in response.events


def test_message_after_termination_reports_stopped() -> None:
    client = new_client()
    sid = client.create_session().session_id
    client.send_message(sid, "我想死。")
    follow_up = client.send_message(sid, "还在吗？")
    assert follow_up.error_code == ErrorCode.SESSION_STOPPED


# --------------------------------------------------------------------- L2
def test_scale_requires_sequential_item_numbers() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    response = client.send_message(sid, "白天也没力气。")
    assert response.waiting_type == "scale_item"

    wrong = client.submit_scale(sid, "5", 1)
    assert wrong.error_code == ErrorCode.SCHEMA_ERROR
    assert "期望第" in wrong.error_message


def test_scale_submission_without_pending_scale_is_rejected() -> None:
    client = new_client()
    sid = client.create_session().session_id
    response = client.submit_scale(sid, "1", 1)
    assert response.error_code == ErrorCode.SCHEMA_ERROR


def test_finishing_scale_turn_is_booked_only_once() -> None:
    """量表答完那一轮由 ``_advance`` 记账，合并措辞时不能再记第二遍。

    ``route_history`` 与 ``l5_reviews`` 是在同一处追加的，所以查一处就够——
    重复记账会让 L5 的逐轮复盘和疗程总结里的 ``route_summary`` 平白多一条。
    """

    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    response = client.send_message(sid, "白天也没力气。")
    response = answer_scales(client, sid, response)

    assert response.waiting_type == "profile_confirmation"
    reviews = client.get_dev_state(sid).layers["L5"]["reviews"]
    turns = [item["turn"] for item in reviews]
    assert len(turns) == len(set(turns)), f"同一轮被记了两次：{turns}"


def test_scales_complete_then_ask_for_profile_confirmation() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    response = client.send_message(sid, "白天也没力气。")

    # 逐份量表作答，把整段过程中的事件都收起来——最后一份量表完成时返回的响应里
    # 只带 `l2_scales_completed`，单份完成事件在中间那一轮。
    seen: list[str] = []
    while response.waiting_type == "scale_item" and response.scale_item is not None:
        seen.extend(response.events)
        item = response.scale_item
        response = client.submit_scale(sid, item.item_number, 1)
    seen.extend(response.events)

    assert response.waiting_type == "profile_confirmation"
    assert "l2_scale_completed" in seen
    assert "l2_scales_completed" in seen
    assert "l2_profile_ready" in seen


def test_profile_modify_does_not_confirm() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    response = answer_scales(
        client, sid, client.send_message(sid, "白天也没力气。")
    )
    assert response.waiting_type == "profile_confirmation"

    modified = client.confirm_profile(sid, ProfileDecision.MODIFY)
    assert modified.waiting_type == "none"
    assert "l2_profile_confirmed" not in modified.events
    state = client.get_dev_state(sid)
    assert state.layers["L2"]["profile_confirmed"] is False

    again = client.send_message(sid, "主要出现在下班以后。")
    assert again.waiting_type == "profile_confirmation", "改完之后要再确认一次"


def test_profile_pause_sets_paused_state() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    response = answer_scales(client, sid, client.send_message(sid, "白天也没力气。"))
    paused = client.confirm_profile(sid, ProfileDecision.PAUSE)
    assert paused.waiting_type == "pause"
    assert "l2_profile_confirmed" not in paused.events


# --------------------------------------------------------------- 悬停 / 恢复
def test_pause_and_resume() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")

    paused = client.set_pause(sid, True)
    assert paused.waiting_type == "pause"

    talk_while_paused = client.send_message(sid, "在吗")
    assert talk_while_paused.waiting_type == "pause"

    resumed = client.set_pause(sid, False)
    assert resumed.waiting_type != "pause"


def test_pause_resume_books_each_turn_once() -> None:
    """暂停/恢复各只记一笔。

    恢复走的是「``_advance`` 记账 + 合并措辞」这条路，和量表答完同一个套路，
    最容易把一轮记两次。暂停不推进轮次，所以轮号本来就会重复——这里数的是
    「一次调用新增了几条逐轮复盘」，而不是轮号唯一性。
    """

    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "情绪特别低落，晚上睡不着。")
    client.send_message(sid, "白天也没力气。")

    def reviews() -> list[dict]:
        return client.get_dev_state(sid).layers["L5"]["reviews"]

    before = len(reviews())
    client.set_pause(sid, True)
    paused = len(reviews())
    assert paused == before + 1, "暂停只该记一笔"

    client.set_pause(sid, False)
    assert len(reviews()) == paused + 1, "恢复只该记一笔"
    assert [item["waiting_type"] for item in reviews()].count("pause") == 1


# --------------------------------------------------------------------- L3
def test_l3_loop_back_when_too_few_symptoms() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    # 只报一个症状：信息不足，L3 应当退回 L2 补充。
    # intake_turns=1 时，要两条消息才会进入量表阶段，所以第二条只是「推进流程」。
    response = drive(client, sid, ["情绪一直很低落。", "白天也没力气。"])
    assert "l3_insufficient_return_l2" in response.events
    assert "l2_supplement_requested" in response.events
    assert response.route.main_layer == "L3"
    assert "L2" in response.route.secondary_layers


def test_bipolar_excluded_when_no_manic_history() -> None:
    """验收场景 4 的契约级验证：易激惹 + 睡眠减少、明确否认躁狂史 → 双相被排除。"""

    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    response = drive(
        client,
        sid,
        ["情绪特别低落，一点小事就发火，晚上睡不着。", "白天也没力气。"],
    )
    # L3 出结论前必须先追问躁狂史。
    assert response.route.main_layer == "L3"
    assert "兴奋" in response.patient_reply

    final = client.send_message(sid, "没有过。没有过特别兴奋、不需要睡觉、话也特别多的时候。")
    assert final.route.main_layer == "L4"
    assert "l3_candidates_generated" in final.events
    assert "l3_candidate_excluded" in final.events

    excluded = client.get_dev_state(sid).layers["L3"]["excluded"]
    codes = [item["code"] for item in excluded]
    assert "D008" in codes and "D009" in codes
    # 抑郁本身不该被牵连排除。
    candidates = [item["code"] for item in client.get_dev_state(sid).layers["L3"]["candidates"]]
    assert "D013" in candidates


def test_denial_wording_is_not_taken_as_mania_evidence() -> None:
    """否认句里的「不需要睡觉、话也特别多」不能被当成躁狂证据。"""

    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    client.send_message(sid, "没有过。没有过特别兴奋、不需要睡觉、话也特别多的时候。")
    state = client.get_dev_state(sid)
    assert state.layers["L2"]["exclusions_assessed"] == []


def test_candidates_carry_evidence_chain() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    final = drive(client, sid, ["还是老样子。"])
    assert final.route.main_layer == "L4"
    assert "l3_evidence_chain_complete" in final.events
    assert "l3_verifier_passed" in final.events
    for candidate in client.get_dev_state(sid).layers["L3"]["candidates"]:
        assert candidate["patient_refs"], "候选必须有患者侧证据"
        assert candidate["knowledge_refs"], "候选必须有知识侧证据"


# --------------------------------------------------------------------- L4/L5
def test_treatment_improvement_terminates() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    drive(client, sid, ["还是老样子。"])

    final = None
    for text in ["好一些了，睡得着了。", "有进步，我轻松不少。", "想试试看。"]:
        final = client.send_message(sid, text)
    assert "l4_improvement_detected" in final.events
    assert final.termination_reason == "completed_improved"
    assert final.session_ended is True


def test_treatment_no_change_terminates() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    drive(client, sid, ["还是老样子。"])

    final = None
    for _ in range(6):
        final = client.send_message(sid, "没什么变化，还是那样。")
    assert "l4_no_change_detected" in final.events
    assert final.termination_reason == "completed_no_change"


def test_treatment_dropout_terminates_immediately() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    drive(client, sid, ["还是老样子。"])

    final = client.send_message(sid, "我觉得先不聊了吧，最近太忙。")
    assert "l4_dropout_detected" in final.events
    assert final.termination_reason == "dropout"
    assert final.session_ended is True

    after = client.send_message(sid, "在吗")
    assert after.error_code == ErrorCode.SESSION_STOPPED


def test_episode_summary_none_until_terminated_then_present() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    drive(client, sid, ["还是老样子。"])

    assert client.get_episode_summary(sid) is None

    client.send_message(sid, "我觉得先不聊了吧。")
    summary = client.get_episode_summary(sid)
    assert summary is not None
    assert summary.termination_reason == "dropout"
    assert summary.session_id == sid
    assert summary.l3_result


def test_episode_summary_includes_the_final_turn() -> None:
    """终止那一轮也要进总结。

    ``_terminate`` 在终止轮被记入 ``route_history`` / ``l5_reviews`` 之前就跑了，
    所以总结必须现算；否则 S03 这种只有一轮的会话，总结会整份都是空的。
    """

    client = new_client()
    sid = client.create_session().session_id
    client.send_message(sid, "我想死。")

    summary = client.get_episode_summary(sid)
    assert summary is not None
    assert summary.turns == 1
    assert len(summary.l5_reviews) == 1, "只有一轮，总结里就该有这一轮"
    assert summary.route_summary == {"L1": 1}


def test_history_records_finished_sessions() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    drive(client, sid, ["还是老样子。"])
    client.send_message(sid, "先不聊了。")

    history = client.get_session_history()
    assert any(item.session_id == sid for item in history)
    assert client.get_session_history(sid)[0].termination_reason == "dropout"


def test_dev_state_exposes_all_five_layers() -> None:
    client = new_client(intake_turns=1)
    sid = client.create_session().session_id
    drive(client, sid, ["情绪特别低落，晚上睡不着，白天也没力气。"])
    state = client.get_dev_state(sid)
    assert {"L1", "L2", "L3", "L4", "L5"} <= set(state.layers)
    assert state.layers["L1"]["risk_level"] == "none"
    assert state.layers["L2"]["symptoms"]


# ------------------------------------------------------------------ 可替换性
def test_latency_is_honoured() -> None:
    """``latency_ms`` 用来在报告里体现「模型耗时」，设了就要生效。

    下界留 5ms 余量：Windows 的睡眠受 15.6ms 计时器粒度影响，``sleep(0.02)``
    实测偶尔差零点几毫秒不到 20ms（冷启动整轮跑时会偶发失败）。
    零延迟的桩远低于这个下界，所以放宽后仍能证明 latency_ms 生效。
    """

    import time

    client = new_client(latency_ms=20.0)
    sid = client.create_session().session_id
    started = time.perf_counter()
    client.send_message(sid, "我最近情绪不太好。")
    assert (time.perf_counter() - started) >= 0.015


def test_two_clients_do_not_share_sessions() -> None:
    first = new_client()
    second = new_client()
    sid = first.create_session().session_id
    with pytest.raises(Exception):
        second.get_dev_state(sid)
