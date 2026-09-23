import json

from layers.layer1 import detector
from layers.layer2 import L2AssessmentService, L2DialogueAdapter, L2ToL3Adapter
from layers.layer3.api import diagnose
from layers.layer4.config import Settings
from layers.layer4.llm.mock import MockBackend
from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.service import L4Service, SessionRequest, TurnRequest
from tests.instruction_test_double import controlled_turn
from tests.l2_test_double import ControlledL2Model
from tests.l3_test_double import controlled_clinicr


def _complete_phq9(adapter: L2DialogueAdapter, session_id: str) -> None:
    texts = [
        "几乎每天没有兴趣。",
        "几乎每天情绪低落。",
        "一半以上时间睡不好。",
        "一半以上时间没有精力。",
        "完全没有食欲变化。",
        "一半以上时间觉得自己让人失望。",
        "完全没有注意力变化。",
        "完全没有动作和说话变化。",
        "完全没有伤害自己或不如死去的想法。",
    ]
    for text in texts:
        outcome = adapter.handle_message(session_id, text)
        assert outcome.accepted is True


def test_l1_to_l5_complete_pipeline(monkeypatch):
    """受控模型响应下，真实运行五层业务逻辑和所有层间守卫。"""
    monkeypatch.setattr(detector, "llm_classify", lambda text, context=None: {
        "llm_score": 1,
        "risk_types": [],
        "confidence": 0.95,
        "evidence_refs": [{
            "source": "L1_controlled_model",
            "finding": "no_immediate_crisis",
            "basis": "integration_test",
        }],
    })
    monkeypatch.setattr(
        "layers.layer5.reviewer.ask_llm",
        lambda prompt, **kwargs: json.dumps({
            "harm": "pass",
            "boundary": "pass",
            "quality": "pass",
            "fail_reason": "",
            "fix_instruction": "",
        }, ensure_ascii=False),
    )

    initial_message = (
        "最近三周我每天都很低落，对以前喜欢的事情也没兴趣，"
        "晚上睡不好而且没有精力，觉得自己让人失望。"
    )

    # L1：规则、受控分类模型、策略执行器和出口守卫均实际运行。
    passport = detector.safety_detect(
        initial_message,
        session_id="five-layer-e2e",
        message_id="initial-message",
    )
    assert passport["layer_id"] == "L1"
    assert passport["guard"]["can_call_l2"] is True
    assert passport["execution"]["route"] == "l2"

    # L2：MI 会话、症状抽取、PHQ-9、画像完整度和证据链实际运行。
    l2 = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    dialogue = L2DialogueAdapter(l2)
    session_id = l2.create_session(passport, session_id="five-layer-e2e")
    first = dialogue.handle_message(session_id, initial_message)
    assert first.l2_turn is not None
    assert first.l2_turn.stage_after == "focusing"
    focused = dialogue.handle_message(
        session_id, "我想先从情绪低落和睡眠问题聊起。"
    )
    assert focused.l2_turn is not None
    assert focused.l2_turn.triggered_scales == []
    explored = dialogue.handle_message(
        session_id,
        "这些低落和睡不好已经明显影响了工作。",
        expected_input="functional_impairment",
    )
    assert explored.l2_turn is not None
    assert explored.l2_turn.triggered_scales == ["PHQ-9"]
    dialogue.handle_message(
        session_id, "我同意。", expected_input="scale_consent"
    )
    _complete_phq9(dialogue, session_id)
    diagnostic_answers = [
        ("mania_or_hypomania_history", "从未有过少睡却不困的时期。"),
            ("substance_or_medical_cause", "这并非药物、物质或身体疾病导致。"),
            ("psychotic_symptoms", "最近没有听到别人听不到的声音，也没有感觉被监视。"),
            ("differential_diagnoses_assessed", "已经核对过双相障碍和适应障碍。"),
        ("evidence_conflicts", "量表结果和我的实际感受一致，没有矛盾。"),
    ]
    for expected_input, answer in diagnostic_answers:
        structured = dialogue.handle_message(
            session_id, answer, expected_input=expected_input
        )
        assert structured.accepted is True
    dialogue.handle_message(
        session_id,
        "我的目标是恢复工作状态和改善睡眠。",
        expected_input="patient_goals",
    )
    dialogue.handle_message(
        session_id,
        "重要性8分，信心6分，准备度7分。",
        expected_input="motivation",
    )
    l2.confirm_profile(session_id, "continue")
    intake = l2.build_clinical_intake(session_id)
    assert intake.layer_id == "L2"
    assert intake.information_sufficiency == "sufficient"

    # L2→L3：守卫验证证据绑定，再转换当前 L3 所需字段。
    l2_to_l3 = L2ToL3Adapter()
    receipt = l2_to_l3.assert_valid(intake)
    profile = l2_to_l3.to_profile(intake)
    assert receipt["route"] == "allow_l3"
    assert profile["PHQ9"] == 12
    assert profile["PHQ9_items"]["item1"] == 3

    # L3：候选诊断、逐条 DSM 证据、独立 Verifier、疗法匹配和出口凭证。
    l3_engine, _, _ = controlled_clinicr()
    l3_raw = diagnose(profile, safety_context={
        "risk_level": passport["risk_level"],
        "sensitive_topics": passport["sensitive_topics"],
        "flags": passport["flags"],
        "l1_flags": passport["flags"],
    }, clinicr=l3_engine)
    report = DiagnosticReport.model_validate(l3_raw)
    assert report.layer_id == "L3"
    assert report.diagnosis.primary is not None
    assert report.diagnosis.primary.disorder == "MDD"
    assert report.verifier_routing == "pass"
    assert report.verifier_score == 0.87
    assert report.l3_validation is not None and report.l3_validation.passed

    # L4→L5：L4 用受控生成后端；L5 使用真实本地规则构造和审核适配器。
    l4 = L4Service(
        config=Settings(llm_provider="mock", review_mode="l5_local"),
        backend=MockBackend(default=json.dumps({
            "reply": "听起来这三周的低落和疲惫已经明显影响了工作。我们可以先一起找一个今天能够完成的小行动。",
            "stage_complete": False,
            "reason": "保持共情并使用行为激活方向",
        }, ensure_ascii=False)),
    )
    l4_session = l4.create_session(SessionRequest(
        session_id="five-layer-e2e",
        report=report,
        safety=SafetyPassport(
            risk_level=passport["risk_level"],
            sensitive_topics=passport["sensitive_topics"],
            flags=passport["flags"],
        ),
    ))
    outcome = l4.handle_turn(TurnRequest(
        session_id=l4_session,
        user_message="我想先从恢复一点工作状态开始。",
        turn=controlled_turn(l4, l4_session, "我想先从恢复一点工作状态开始。"),
    ))

    assert outcome.fallback_used is False
    assert outcome.l5_context.l4_raw_output == outcome.reply
    assert outcome.review_history
    assert outcome.review_history[-1].passed is True
    assert outcome.summary.layer_id == "L4"
