"""病例外事实和发送前阻断的回归；不调用真实 API。"""
import json

import pytest

from patient_agent.generator import PatientAction
from patient_agent.guard import ConsistencyGuard, GuardCode
from patient_agent.library import ScenarioLibrary
from patient_agent.semantic_guard import SemanticFactVerifier
from patient_agent.client.base import ScaleItemPrompt


def say(text, fact_id="", is_denial=False):
    return PatientAction(kind="say", text=text, fact_id=fact_id, is_denial=is_denial)


@pytest.mark.parametrize("text", [
    "医生，那个药就先别管了吧。我最近老觉得，死了算了，活着真没什么意思。",
    "那阵子其实挺短的。后来一下子掉下来了，情绪一直低落，差不多有一个月了。",
    "我没有工作压力，但我想死。",
])
def test_reported_hallucinations_blocked(project_root, text):
    scenario = ScenarioLibrary(project_root / "imported_scenarios").load("DB_P0001")
    assert not ConsistencyGuard(scenario).check(say(text)).ok


def test_valid_fragment_does_not_cover_added_drug(guard_factory):
    scenario, guard = guard_factory("S01_low_risk_depression")
    fact = next(f for f in scenario.revealable_facts() if f.fact_id == "background:bg_duration")
    assert not guard.check(say(fact.utterance + "我也吃了安眠药。", fact.fact_id)).ok


def test_fake_fact_and_chinese_duration(guard_factory):
    _, guard = guard_factory("S01_low_risk_depression")
    assert "UNKNOWN_FACT" in guard.check(say("我在听。", "fake:id")).codes()
    assert "TIMELINE_CONFLICT" in guard.check(say("有一个月了。", "background:bg_duration")).codes()
    assert guard.check(say("两个多月了，大概是换工作之后慢慢开始的。", "background:bg_duration")).ok


def test_denial_flag_cannot_hide_positive_risk(guard_factory):
    _, guard = guard_factory("S01_low_risk_depression")
    assert "RISK_ESCALATION" in guard.check(say("没有工作压力，但想死。", "risk:absent", True)).codes()


def test_non_say_score_must_match_case(guard_factory):
    _, guard = guard_factory("S01_low_risk_depression")
    item = ScaleItemPrompt(scale_name="PHQ-9", item_number="1", prompt="题目", min_score=0, max_score=3)
    assert not guard.check(PatientAction(kind="submit_scale", item_number="2", score=2), scale_item=item).ok
    assert not guard.check(PatientAction(kind="submit_scale", item_number="1", score=1), scale_item=item).ok


def test_locked_fact_cannot_be_disclosed(guard_factory):
    scenario, _ = guard_factory("S01_low_risk_depression")
    fact_id = "background:bg_duration"
    background = next(f for f in scenario.background.facts if f.fact_id == "bg_duration")
    background.unlock_after_events = ["required_event"]
    guard = ConsistencyGuard(scenario)
    reference = next(f.utterance for f in scenario.revealable_facts() if f.fact_id == fact_id)
    assert "DISCLOSURE_VIOLATION" in guard.check(say(reference, fact_id), events=[]).codes()


class ReviewModel:
    def __init__(self, status="supported", partial=False, fail=False):
        self.status, self.partial, self.fail = status, partial, fail

    def complete(self, **kwargs):
        if self.fail:
            raise TimeoutError()
        data = json.loads(kwargs["user"])
        text = data["patient_answer"]
        return json.dumps({"claims": [{"quote": text[:2] if self.partial else text,
            "status": self.status, "reference_ids": list(data["references"])[:1] if self.status == "supported" else [],
            "reason": "测试复核"}]}, ensure_ascii=False)


@pytest.mark.parametrize("status,partial,fail,expected", [
    ("supported", False, False, True), ("unsupported", False, False, False),
    ("contradicted", False, False, False), ("supported", True, False, False),
    ("supported", False, True, False),
])
def test_semantic_review_contract(guard_factory, status, partial, fail, expected):
    scenario, _ = guard_factory("S01_low_risk_depression")
    guard = ConsistencyGuard(scenario, verifier=SemanticFactVerifier(ReviewModel(status, partial, fail)))
    result = guard.check(say("换工作之后逐渐开始的。", "background:bg_duration"))
    assert result.ok == expected
    if partial or fail:
        assert GuardCode.GUARD_REVIEW_UNAVAILABLE.value in result.codes()
        # 报告里必须能看出「为什么复核不可用」，否则会把复核环节的问题误读成患者编造。
        message = result.violations[0].message
        assert ("未完整覆盖回答原文" in message) if partial else ("TimeoutError" in message)


@pytest.mark.parametrize("bad_opening", [True, False])
def test_runner_never_sends_rejected_text(runner_factory, monkeypatch, bad_opening):
    stack = runner_factory("S01_low_risk_depression")
    stack.runner.options.max_retries = 0
    stack.scenario.stop.stop_on_violation = False
    sent = []
    original = stack.client.send_message
    def capture(session_id, text):
        sent.append(text)
        return original(session_id, text)
    monkeypatch.setattr(stack.client, "send_message", capture)
    bad = say("我正在服用不存在的药物。", "opening" if bad_opening else "")
    monkeypatch.setattr(stack.generator, "opening" if bad_opening else "next_action", lambda *args: bad)
    recorder = stack.runner.run()
    assert bad.text not in sent
    assert len(sent) == (0 if bad_opening else 1)
    assert not recorder.turns[-1].submitted
    assert recorder.turns[-1].patient_text == ""
    assert recorder.turns[-1].rejected_text == bad.text


def test_retry_recovers_from_trusted_reference(runner_factory, monkeypatch):
    stack = runner_factory("S01_low_risk_depression")
    action = stack.generator.opening()
    action.text = "我最近服用了未知药物。"
    corrected, result, retries = stack.runner.validate_action(action)
    assert result.ok and retries == 1
    assert "未知药物" not in corrected.text
