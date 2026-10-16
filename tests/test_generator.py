"""PA3：患者生成器。

断言的都是「病例脚本 → 台词」这条链上的确定性行为：说了什么、什么时候说、
被追问才说的否认、量表按计划给分、轨迹按治疗轮次推进。
"""

from __future__ import annotations

from patient_agent.client.base import ScaleItemPrompt
from patient_agent.errors import LLMError
from patient_agent.generator import PatientAction, PatientGenerator, TurnContext
from patient_agent.lexicon import DIAGNOSIS_LABELS
from patient_agent.llm import RuleBasedLLM


def probe(text: str, *, layer: str = "L2", treatment_turn: int = 0, events=None) -> TurnContext:
    return TurnContext(
        last_reply=text,
        main_layer=layer,
        treatment_turn=treatment_turn,
        events=list(events or []),
    )


# --------------------------------------------------------------------- 开场
def test_opening_does_not_duplicate_chief_complaint(generator_factory) -> None:
    """S03 的风险台词与主诉措辞接近，开场不该把「想死」贴两遍。"""

    _, generator = generator_factory("S03_explicit_crisis")
    text = generator.opening().text
    assert text.count("想死") == 1
    assert "不想活了" not in text


def test_opening_appends_chief_complaint_when_missing(generator_factory) -> None:
    scenario, generator = generator_factory("S02_insufficient_info")
    text = generator.opening().text
    assert scenario.clinical.chief_complaint in text


def test_opening_is_deterministic_across_reset(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    first = generator.opening().text
    generator.reset()
    assert generator.opening().text == first


# --------------------------------------------------------------- 事实披露
def test_probe_keyword_selects_matching_fact(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    assert action.kind == "say"
    assert action.fact_id == "background:bg_duration"
    assert "2个多" in action.text


def test_each_fact_is_disclosed_at_most_once(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    first = generator.next_action(probe("这种情况大概持续多久了？"))
    assert first.fact_id == "background:bg_duration"
    # 再问同一句：那条已经说过，应该换别的或敷衍，而不是重复。
    second = generator.next_action(probe("这种情况大概持续多久了？"))
    assert second.fact_id != "background:bg_duration"


def test_denial_requires_probe(generator_factory) -> None:
    """否认类事实不能被主动说出来，必须被追问。"""

    _, generator = generator_factory("S04_bipolar_exclusion")
    # 一句和躁狂史无关的话：不能蹦出「没有过特别兴奋」。
    action = generator.next_action(probe("今天天气不错"))
    assert "特别兴奋" not in action.text

    denial = generator.next_action(probe("有没有过一段时间特别兴奋、不需要睡觉、话也特别多？"))
    assert denial.is_denial is True
    assert denial.fact_id == "exclusion_absent:manic_history"
    assert denial.text.startswith("没有过")


def test_risk_denial_requires_probe(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("我想认真问一句：有没有出现过伤害自己的念头？"))
    assert action.is_denial is True
    assert action.fact_id == "risk:absent"


def test_unlocked_fact_waits_for_event(generator_factory) -> None:
    """S02 的补充事实挂在 L3→L2 回环事件上，事件没出现就不该说。

    用的追问语只命中那条被锁的事实（「缓不过来」），否则会先被别的症状抢走，
    断言就测不到闸门本身了。
    """

    _, generator = generator_factory("S02_insufficient_info")
    gated = [
        fact
        for fact in generator._facts  # noqa: SLF001 - 单测需要看清闸门
        if "l3_insufficient_return_l2" in fact.unlock_after_events
    ]
    assert gated, "S02 应当有被事件解锁的事实"
    gated_id = gated[0].fact_id

    # 未观测到事件：即便被追问到也只字不提。
    generator.next_action(probe("最近总觉得缓不过来。"))
    assert gated_id not in generator.state.disclosed

    # 观测到事件后再追问，就会说。
    generator.next_action(probe("最近总觉得缓不过来。", events=["l3_insufficient_return_l2"]))
    assert gated_id in generator.state.disclosed


# --------------------------------------------------------------- 画像确认
def test_profile_decision_follows_plan_order(generator_factory) -> None:
    """S06 先要求修改，再确认。"""

    _, generator = generator_factory("S06_treatment_dropout")
    ctx = TurnContext(waiting_type="profile_confirmation")
    first = generator.next_action(ctx)
    second = generator.next_action(ctx)
    assert (first.kind, first.decision) == ("confirm_profile", "modify")
    assert (second.kind, second.decision) == ("confirm_profile", "continue")


def test_default_profile_decision_is_continue(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(TurnContext(waiting_type="profile_confirmation"))
    assert action.decision == "continue"


# ----------------------------------------------------------------- 量表
def test_scale_answers_follow_plan(generator_factory) -> None:
    scenario, generator = generator_factory("S01_low_risk_depression")
    plan = scenario.clinical.scale_plans[0]
    for index, expected in enumerate(plan.answers):
        item = ScaleItemPrompt(
            scale_name=plan.scale_name,
            item_number=str(index + 1),
            prompt="题面",
            min_score=0,
            max_score=3,
        )
        action = generator.next_action(
            TurnContext(waiting_type="scale_item", scale_item=item)
        )
        assert action.kind == "submit_scale"
        assert action.score == expected
        assert action.item_number == str(index + 1)


def test_scale_score_is_clamped_to_item_range(generator_factory) -> None:
    """病例给的分数超过题目上限时必须截断，否则系统会判为非法作答。"""

    _, generator = generator_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=2
    )
    action = generator.next_action(TurnContext(waiting_type="scale_item", scale_item=item))
    assert action.score == 2


def test_unknown_scale_answers_zero(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(
        scale_name="Y-BOCS", item_number="1", prompt="题面", min_score=0, max_score=4
    )
    action = generator.next_action(TurnContext(waiting_type="scale_item", scale_item=item))
    assert action.score == 0, "病例没计划的量表不能编造分数"


def test_refuse_scale_says_refusal_not_a_score(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    generator.scenario.clinical.scale_plans[0].willingness = "refuse"
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=3
    )
    action = generator.next_action(TurnContext(waiting_type="scale_item", scale_item=item))
    assert action.kind == "say"
    assert action.text


# ----------------------------------------------------------------- 轨迹
def test_trajectory_step_gated_by_treatment_turn(generator_factory) -> None:
    """轨迹步按「治疗轮次」推进，量表逐题作答不该把它提前。"""

    _, generator = generator_factory("S01_low_risk_depression")
    # 还没进治疗层：不该出轨迹台词。
    early = generator.next_action(probe("你愿意说说最近具体是什么让你这么难受吗？"))
    assert not early.fact_id.startswith("trajectory:")

    first_step = generator.next_action(
        TurnContext(main_layer="L4", treatment_turn=1, last_reply="这一周怎么样？")
    )
    assert first_step.fact_id == "trajectory:0"


def test_trajectory_runs_out_gracefully(generator_factory) -> None:
    scenario, generator = generator_factory("S01_low_risk_depression")
    for index in range(len(scenario.trajectory.steps)):
        action = generator.next_action(
            TurnContext(main_layer="L4", treatment_turn=index + 1, last_reply="")
        )
        assert action.fact_id == f"trajectory:{index}"
    # 轨迹用尽后仍然给得出话（落到事实或敷衍），不能空转。
    last = generator.next_action(TurnContext(main_layer="L4", treatment_turn=99, last_reply=""))
    assert last.kind == "say"
    assert last.text


# --------------------------------------------------------------- 生命周期
def test_session_ended_yields_stop(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(
        TurnContext(session_ended=True, termination_reason="completed_improved")
    )
    assert action.kind == "stop"
    assert "completed_improved" in action.reason


def test_pause_resumes_when_planned(generator_factory) -> None:
    _, generator = generator_factory("S05_comorbid_anxiety_depression")
    action = generator.next_action(TurnContext(waiting_type="pause"))
    assert action.kind == "set_pause"
    assert action.paused is False


def test_pause_stops_when_resume_disabled(generator_factory) -> None:
    scenario, generator = generator_factory("S01_low_risk_depression")
    scenario.pause_plan.resume_after_pause = False
    action = generator.next_action(TurnContext(waiting_type="pause"))
    assert action.kind == "stop"


def test_reset_clears_progress(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    generator.next_action(probe("这种情况大概持续多久了？"))
    assert generator.state.disclosed
    generator.reset()
    assert generator.state.disclosed == []
    assert generator.state.profile_asked == 0


def test_refusal_utterance_comes_from_scenario(generator_factory) -> None:
    scenario, generator = generator_factory("S01_low_risk_depression")
    generator.scenario.clinical.scale_plans[0].willingness = "refuse"
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=3
    )
    action = generator.next_action(TurnContext(waiting_type="scale_item", scale_item=item))
    assert scenario.disclosure.refusal_utterances[0] in action.text


# --------------------------------------------------------------- 措辞层
def test_offline_renderer_returns_reference_line(generator_factory) -> None:
    """离线模式下渲染结果应当等于参考台词（确定性）。"""

    scenario, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    reference = generator._reference_line(action.prompt_user)  # noqa: SLF001
    assert action.text == reference


def test_prompt_lists_never_disclose_items(generator_factory) -> None:
    scenario, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    for item in scenario.disclosure.never_disclose:
        assert item in action.prompt_system


def test_prompt_forbids_internal_jargon(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    assert "L1~L5" in action.prompt_system
    assert "系统提示" in action.prompt_system


def test_prompt_carries_summary_and_history(generator_factory) -> None:
    """方案二：摘要与历史必须进 prompt，且诊断名不得被一并带进去。

    这层断言跑在离线适配器上——被测产物是 prompt 本身，不是模型对它的反应，
    所以「上下文增强」在 CI 里依然可回归。
    """

    scenario, generator = generator_factory("S01_low_risk_depression")
    context = TurnContext(
        last_reply="你觉得是什么让你这样？",
        main_layer="L2",
        recent_turns=[("这种情况大概持续多久了？", "三个月了。")],
    )
    action = generator.next_action(context)
    assert scenario.clinical.chief_complaint in action.prompt_system
    assert "医生：这种情况大概持续多久了？" in action.prompt_user
    assert "我：三个月了。" in action.prompt_user
    combined = action.prompt_system + action.prompt_user
    assert not any(label in combined for label in DIAGNOSIS_LABELS)


def test_retry_keeps_same_fact(generator_factory) -> None:
    """守卫打回后重说，必须还是同一条事实，不能换成别的内容。"""

    _, generator = generator_factory("S01_low_risk_depression")
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    retried = generator.retry_action(action)
    assert retried.fact_id == action.fact_id
    assert retried.is_denial == action.is_denial


def test_action_to_send_text_uses_scale_answer(generator_factory) -> None:
    _, generator = generator_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=3
    )
    action = generator.next_action(TurnContext(waiting_type="scale_item", scale_item=item))
    assert generator.action_to_send_text(action) == action.answer_text


class _BrokenLLM(RuleBasedLLM):
    """总抛错的模型，用来验证降级路径。"""

    name = "broken"

    def complete(self, *, system: str, user: str, temperature=None, max_tokens=None) -> str:
        raise LLMError("模型挂了")


def test_llm_failure_falls_back_to_reference(generator_factory) -> None:
    """模型不可用时退回参考台词，会话不能因此崩掉。"""

    _, generator = generator_factory("S01_low_risk_depression", llm=_BrokenLLM())
    action: PatientAction = generator.next_action(probe("这种情况大概持续多久了？"))
    assert action.kind == "say"
    assert "2个多" in action.text


def test_empty_llm_output_falls_back_to_reference(generator_factory) -> None:
    """模型返回空白也算失败，必须回退，不能把空串发给系统。"""

    class _BlankLLM(RuleBasedLLM):
        def complete(self, *, system: str, user: str, temperature=None, max_tokens=None) -> str:
            return "   "

    _, generator = generator_factory("S01_low_risk_depression", llm=_BlankLLM())
    action = generator.next_action(probe("这种情况大概持续多久了？"))
    assert action.text.strip()
    assert "2个多" in action.text
