"""PA4：一致性守卫。

守的是任务说明里那条硬约束——「Patient Agent 不会添加病例外事实」。
所以每个用例都在问同一个问题：这句话，病例允不允许？
"""

from __future__ import annotations

from patient_agent.client.base import ScaleItemPrompt
from patient_agent.generator import PatientAction
from patient_agent.guard import GuardCode


def say(text: str, *, fact_id: str = "", is_denial: bool = False) -> PatientAction:
    return PatientAction(
        kind="say", text=text, fact_id=fact_id, is_denial=is_denial, reason="单测"
    )


def codes(result) -> list[str]:
    return result.codes()


# ------------------------------------------------------------------ 基本形态
def test_clean_fact_utterance_passes(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(
        say("2个多月了，大概是换工作之后慢慢开始的。", fact_id="background:bg_duration")
    )
    assert result.ok is True
    assert result.violations == []


def test_empty_utterance_is_blocking(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("   "))
    assert result.ok is False
    assert GuardCode.EMPTY_UTTERANCE.value in codes(result)
    assert result.blocking


def test_non_say_actions_are_never_blocked(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    for action in (
        PatientAction(kind="confirm_profile", decision="continue"),
        PatientAction(kind="set_pause", paused=True),
        PatientAction(kind="stop"),
    ):
        assert guard.check(action).ok is True


# ------------------------------------------------------------------ 结构泄露
def test_internal_jargon_is_blocked(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("我听说这个系统的 L3 会给出置信度。"))
    assert result.ok is False
    assert GuardCode.INTERNAL_JARGON_LEAK.value in codes(result)


def test_plain_patient_speech_has_no_jargon(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    assert GuardCode.INTERNAL_JARGON_LEAK.value not in codes(guard.check(say("我就是晚上睡不好，白天没力气。")))


# ------------------------------------------------------------------ 诊断越界
def test_self_diagnosis_is_blocked(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("我觉得我就是抑郁症。"))
    assert result.ok is False
    assert GuardCode.DIAGNOSIS_OVERREACH.value in codes(result)


def test_attributed_diagnosis_is_allowed(guard_factory) -> None:
    """转述别人的话不算患者自己下结论。"""

    _, guard = guard_factory("S01_low_risk_depression")
    assert GuardCode.DIAGNOSIS_OVERREACH.value not in codes(guard.check(say("社区医生说我是抑郁症，我自己不太确定。")))


def test_patient_reported_label_is_allowed(guard_factory) -> None:
    """S04 授权患者说出「双相障碍」，这是「不迎合」验证的台词。"""

    _, guard = guard_factory("S04_bipolar_exclusion")
    assert GuardCode.DIAGNOSIS_OVERREACH.value not in codes(guard.check(say("社区医生提过双相障碍，但我不觉得是。")))


def test_unrelated_label_still_blocked_even_with_other_labels_allowed(guard_factory) -> None:
    _, guard = guard_factory("S04_bipolar_exclusion")
    result = guard.check(say("我怀疑自己是精神分裂症。"))
    assert result.ok is False
    assert GuardCode.DIAGNOSIS_OVERREACH.value in codes(result)


# ------------------------------------------------------------ 绝不披露 / 矛盾
def test_never_disclose_is_blocked(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("我已经写好了遗书。"))
    assert result.ok is False
    assert GuardCode.NEVER_DISCLOSE.value in codes(result)


def test_forbidden_claim_is_blocked(guard_factory) -> None:
    scenario, guard = guard_factory("S01_low_risk_depression")
    claim = scenario.background.forbidden_claims[0]
    result = guard.check(say(f"其实{claim}。"))
    assert result.ok is False
    assert GuardCode.FORBIDDEN_CLAIM.value in codes(result)


# -------------------------------------------------------------------- 数字
def test_fabricated_duration_is_blocking(guard_factory) -> None:
    """编出一个新病程是最危险的一类漂移，必须阻断。"""

    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("我这8个月一直这样。", fact_id="background:bg_duration"))
    assert result.ok is False
    assert GuardCode.FABRICATED_NUMBER.value in codes(result)
    assert result.blocking


def test_declared_number_is_fine(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("2个多月了，大概是换工作之后慢慢开始的。", fact_id="background:bg_duration"))
    assert result.ok is True


def test_timeline_conflict_is_detected(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("我这6个月都这样。"))
    assert GuardCode.TIMELINE_CONFLICT.value in codes(result)


# -------------------------------------------------------------------- 风险
def test_risk_escalation_is_blocked(guard_factory) -> None:
    """低风险病例的患者不会突然说出自伤意念。"""

    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("有时候我想死。"))
    assert result.ok is False
    assert GuardCode.RISK_ESCALATION.value in codes(result)


def test_risk_words_inside_denial_are_ignored(guard_factory) -> None:
    scenario, guard = guard_factory("S01_low_risk_depression")
    # 直接用病例里的否认句，且标成否认。
    fact = next(
        item
        for item in scenario.revealable_facts()
        if item.fact_id == "risk:absent"
    )
    result = guard.check(
        say(fact.denial_utterance, fact_id="risk:absent", is_denial=True)
    )
    assert GuardCode.RISK_ESCALATION.value not in codes(result)


def test_denial_detected_without_flag_is_not_escalation(guard_factory) -> None:
    """就算忘了标 is_denial，句首的「没有过」也应被识别成否认。"""

    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("没有过。没有过伤害自己的念头。"))
    assert GuardCode.RISK_ESCALATION.value not in codes(result)


def test_matching_risk_level_passes(guard_factory) -> None:
    _, guard = guard_factory("S03_explicit_crisis")
    assert GuardCode.RISK_ESCALATION.value not in codes(guard.check(say("我想死，撑不下去了。")))


def test_risk_downgrade_is_not_a_violation(guard_factory) -> None:
    """说得比脚本轻不构成「添加病例外事实」。"""

    _, guard = guard_factory("S03_explicit_crisis")
    assert GuardCode.RISK_ESCALATION.value not in codes(guard.check(say("最近有点累，睡不好。")))


# ------------------------------------------------------------ 事实 / 严重度
def test_fact_mismatch_is_blocking(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(
        say("我上周把家里的旧沙发换成新的了。", fact_id="symptom:depressed_mood")
    )
    assert GuardCode.FACT_MISMATCH.value in codes(result)
    assert result.ok is False


def test_severity_inflation_is_recorded(guard_factory) -> None:
    """S01 的疲乏是轻度（severity=1），说成「每天」要留痕。"""

    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(say("每天都特别累，一直没力气。", fact_id="symptom:fatigue"))
    assert GuardCode.SEVERITY_INFLATION.value in codes(result)
    assert result.ok is False


# -------------------------------------------------------------------- 量表
def test_scale_score_out_of_range_is_blocking(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=3
    )
    result = guard.check(
        PatientAction(kind="submit_scale", item_number="1", score=7), scale_item=item
    )
    assert result.ok is False
    assert GuardCode.SCALE_MISMATCH.value in codes(result)


def test_scale_score_in_range_passes(guard_factory) -> None:
    _, guard = guard_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(
        scale_name="PHQ-9", item_number="1", prompt="题面", min_score=0, max_score=3
    )
    result = guard.check(
        PatientAction(kind="submit_scale", item_number="1", score=2), scale_item=item
    )
    assert result.ok is True


def test_submit_without_item_info_is_blocked(guard_factory) -> None:
    """无法判定时不得提交未经验证的量表分值。"""

    _, guard = guard_factory("S01_low_risk_depression")
    result = guard.check(PatientAction(kind="submit_scale", item_number="1", score=3))
    assert result.ok is False


def test_allow_severity_upgrade_switch(guard_factory) -> None:
    from patient_agent.guard import ConsistencyGuard

    scenario, _ = guard_factory("S01_low_risk_depression")
    lenient = ConsistencyGuard(scenario, allow_severity_upgrade=True)
    result = lenient.check(say("每天都特别累，一直没力气。", fact_id="symptom:fatigue"))
    assert GuardCode.SEVERITY_INFLATION.value not in codes(result)
