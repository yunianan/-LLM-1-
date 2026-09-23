"""L5 上下文、矛盾判定和人工复核路由回归；API 使用受控替身。"""

import json

import pytest

from layers.layer4.review.l5_adapter import L5Reviewer
from layers.layer4.schemas.l5_context import L5Context
from layers.layer4.schemas.review import ReviewResult
from layers.layer4.service import TurnRequest
from layers.layer5.reviewer import review_response
from tests.instruction_test_double import controlled_turn
from tests.test_service import _report
from tests.test_top_level_orchestrator import _orchestrator, _safe_passport


PASS = dict(harm="pass", boundary="pass", quality="pass",
            fail_reason="", fix_instruction="")


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("orchestration.orchestrator.safety_detect", _safe_passport)
    pipeline = _orchestrator(tmp_path)
    pipeline.create_session("l5-test")
    session = pipeline.store.get("l5-test")
    pipeline._start_l4(session, _report())
    pipeline.l4.reviewer = L5Reviewer(review_response)
    return pipeline


def test_real_l5_prompt_receives_current_message_history_and_full_instructions(pipeline, monkeypatch):
    prompts = []

    def api(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(PASS)

    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", api)
    service = pipeline.l4
    sd = service.store.get("l5-test")
    sd.anchor.contraindications = ["严重解离时禁止暴露"]
    sd.safety.flags = ["monitor_suicidal_ideation"]
    message = "邮件之后很难受"
    first_turn = controlled_turn(service, "l5-test", message,
                                 plugin_steps={"CBT-I": "先讨论睡眠习惯记录"})
    first = service.handle_turn(TurnRequest(session_id="l5-test", user_message=message, turn=first_turn))
    message2 = "我昨晚仍然很难入睡"
    second_turn = controlled_turn(service, "l5-test", message2)
    second = service.handle_turn(TurnRequest(session_id="l5-test", user_message=message2, turn=second_turn))

    def context(prompt):
        return json.loads(prompt.split("审核上下文（JSON）：\n", 1)[1].split("\n\n待审核AI回复：", 1)[0])

    first_data, second_data = map(context, prompts)
    assert first_data["user_message"] == message
    assert first_data["history"] == []
    assert first_data["instructions"] == sd.instructions().model_copy(update={"L3": first_turn}).model_dump(mode="json")
    assert first_data["instructions"]["L2"]["contraindications"] == ["严重解离时禁止暴露"]
    assert first_data["instructions"]["L3"]["plugin_steps"] == {"CBT-I": "先讨论睡眠习惯记录"}
    assert first_data["risk_context"]["flags"] == ["monitor_suicidal_ideation"]
    assert second_data["user_message"] == message2
    assert second_data["history"] == [dict(role="user", content=message), dict(role="assistant", content=first.reply)]
    assert second_data["instructions"]["L3"] == second_turn.model_dump(mode="json")
    assert first.l5_context.history == []  # 不被后续持久化反向改写。
    assert len(second.l5_context.history) == 2

    # 外部重写同一轮时，不能把本轮旧回复当作既往对话。
    service.handle_turn(TurnRequest(session_id="l5-test", user_message=message2, turn=second_turn),
                        initial_fix="重新核对措辞", persist=False)
    assert context(prompts[-1])["history"] == second_data["history"]


def test_l5_refuses_incomplete_context_before_api(monkeypatch):
    def unexpected(_prompt, **_kwargs):
        pytest.fail("缺少审核上下文时不应请求模型")

    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", unexpected)
    with pytest.raises(ValueError):
        review_response(dict(l4_raw_output="回复", l4_substage="L4-L1", therapy_type="CBT"))


@pytest.mark.parametrize("feedback", [
    {"fail_reason": "回复忽略了患者拒绝暴露"},
    {"fix_instruction": "删除暴露建议"},
    {"fail_reason": "存在越界", "fix_instruction": "删除越界内容"},
])
def test_pass_with_failure_feedback_is_rejected(feedback):
    result = ReviewResult(**(PASS | feedback))
    assert not result.passed
    assert "fail" in (result.harm, result.boundary, result.quality)
    assert "矛盾" in result.fail_reason
    assert result.fix_instruction.strip()


def test_blank_feedback_does_not_reject_valid_pass():
    assert ReviewResult(**(PASS | {"fail_reason": " \n", "fix_instruction": "\t"})).passed


@pytest.mark.parametrize("failure_mode", ["contradiction", "timeout", "reject"])
def test_exhausted_review_pauses_pipeline_and_keeps_l1_active(pipeline, monkeypatch, failure_mode):
    calls = []

    def api(prompt, **kwargs):
        calls.append(prompt)
        if failure_mode == "timeout":
            raise TimeoutError("unavailable")
        if failure_mode == "contradiction":
            return json.dumps(PASS | {"fail_reason": "内部审核指出越界", "fix_instruction": "修正越界"})
        return json.dumps(PASS | {"boundary": "fail", "fail_reason": "内部审核指出越界", "fix_instruction": "修正越界"})

    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", api)
    first = pipeline.handle_message("l5-test", "工作压力大")
    assert first.route == "MANUAL_REVIEW"
    assert first.waiting_for == "manual_review"
    assert len(calls) == 3
    assert "暂停" in first.reply and "复核" in first.reply
    assert "内部审核" not in first.reply
    snapshot = pipeline.get_session_snapshot("l5-test")
    assert "l5_manual_review_required" in snapshot.internal_flags
    assert len(snapshot.pending_l5_review["review_history"]) == 3
    assert "pending_l5_review" not in first.model_dump()

    second = pipeline.handle_message("l5-test", "继续聊聊")
    assert second.route == "MANUAL_REVIEW"
    assert len(calls) == 3
    assert len(pipeline.instruction_planner.model.calls) == 1
    with pytest.raises(ValueError):
        pipeline.resume_session("l5-test")

    crisis = pipeline.handle_message("l5-test", "立即危机")
    assert crisis.route == "STOPPED" and crisis.terminal
    assert "急救" in crisis.reply
    assert len(calls) == 3


def test_failed_then_passed_review_does_not_pause(pipeline, monkeypatch):
    responses = iter([PASS | {"fail_reason": "需要修改", "fix_instruction": "保持共情"}, PASS])
    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", lambda prompt, **kwargs: json.dumps(next(responses)))
    result = pipeline.handle_message("l5-test", "最近工作压力大")
    assert result.route == "L4"
    assert result.waiting_for == "patient_message"
    assert pipeline.get_session_snapshot("l5-test").pending_l5_review is None
    assert pipeline.l4.store.get("l5-test").fallback_history == [False]


@pytest.mark.parametrize("persist", [True, False])
def test_direct_l4_cannot_bypass_pending_manual_review(pipeline, monkeypatch, persist):
    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", lambda prompt, **kwargs: json.dumps(PASS | {"harm": "fail"}))
    pipeline.handle_message("l5-test", "工作压力大")
    sd = pipeline.l4.store.get("l5-test")
    original = sd.treatment_records()[0]
    next_turn = controlled_turn(pipeline.l4, "l5-test", "继续") if persist else original.instruction
    with pytest.raises(ValueError, match="人工复核"):
        pipeline.l4.handle_turn(TurnRequest(session_id="l5-test", turn=next_turn,
                                            user_message="继续" if persist else original.user_message), persist=persist)


def test_reviewer_adapter_rejects_contradictory_external_verdict(pipeline, monkeypatch):
    monkeypatch.setattr("layers.layer5.reviewer.ask_llm", lambda prompt, **kwargs: json.dumps(PASS))
    turn = controlled_turn(pipeline.l4, "l5-test", "压力大")
    out = pipeline.l4.handle_turn(TurnRequest(session_id="l5-test", user_message="压力大", turn=turn))
    ctx = L5Context.model_validate(out.l5_context.model_dump())
    result = L5Reviewer(lambda payload: PASS | {"fix_instruction": "删除未经授权的内容"}).review(ctx, [])
    assert not result.passed
