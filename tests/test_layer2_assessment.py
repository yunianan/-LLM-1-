from __future__ import annotations

from datetime import datetime, timezone

import pytest

from layers.layer2 import (
    L2AssessmentService,
    L2DialogueAdapter,
    L2EntranceGuardError,
    L2ExitGuardError,
    L2ToL3Adapter,
)
from layers.layer2.llm import L2ModelUnavailable
from layers.layer2.profile import ProfileIngestor
from layers.layer2.profile.sufficiency import missing_dimensions
from layers.layer2.scales import SCALE_CATALOG, ScaleEngine
from layers.layer2.schemas import (
    ClinicalIntake,
    L2SessionState,
    ScaleResult,
    SemanticAnalysis,
    SemanticCandidate,
    StageDecision,
    StructuredFieldCandidate,
)
from tests.l2_test_double import ControlledL2Model


def safe_passport(session_id: str = "l2-flow", **overrides) -> dict:
    passport = {
        "schema_version": "1.0",
        "layer_id": "L1",
        "session_id": session_id,
        "message_id": "test-message",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "risk_level": 1,
        "safety_flag": "pass",
        "crisis_flag": False,
        "decision_path": "normal",
        "security_event": {
            "detected": False,
            "event_type": "",
            "severity": 0,
            "matched_patterns": [],
        },
        "sensitive_topics": [],
        "flags": [],
        "structured_labels": {
            "scale_skip_list": [],
            "asq": {"status": "negative"},
            "bssa": {"status": "not_started"},
            "l2_handoff": {
                "ordinary_care_allowed": True,
                "baseline_asq_status": "negative",
            },
        },
        "execution": {
            "route": "l2",
            "terminal": False,
            "applied_controls": [],
        },
        "guard": {"status": "pass", "can_call_l2": True},
    }
    structured_override = overrides.pop("structured_labels", None)
    passport.update(overrides)
    if structured_override is not None:
        passport["structured_labels"].update(structured_override)
    if "decision_path" not in overrides:
        if passport.get("crisis_flag") is True:
            passport["decision_path"] = "clinical_crisis"
        elif "require_safety_followup" in passport.get("flags", []):
            passport["decision_path"] = "safety_followup"
        elif passport.get("security_event", {}).get("detected") is True:
            passport["decision_path"] = "security_reject"
    return passport


def test_information_sufficiency_requires_shared_l2_l3_diagnostic_fields():
    state = L2SessionState(
        session_id="shared-contract",
        safety_passport=safe_passport("shared-contract"),
    )
    state.diagnostic_context.symptom_duration_days = 21
    state.diagnostic_context.functional_impairment = True

    missing = missing_dimensions(state)

    assert "躁狂或轻躁狂史" in missing
    assert "物质、药物或躯体疾病影响" in missing
    assert "精神病性症状筛查" in missing


def build_ready_session():
    model = ControlledL2Model()
    service = L2AssessmentService(model=model, require_fresh_passport=False)
    adapter = L2DialogueAdapter(service)
    sid = service.create_session(safe_passport(), session_id="l2-flow")

    first = adapter.handle_message(
        sid,
        "最近三周我每天都很低落，对以前喜欢的事情没兴趣，晚上睡不好，"
        "也没有精力，觉得自己让人失望。",
    )
    assert first.l2_turn and first.l2_turn.stage_after == "focusing"

    focus = adapter.handle_message(sid, "我想先从情绪低落和睡眠问题聊起。")
    assert focus.l2_turn and focus.l2_turn.triggered_scales == []
    assert focus.awaiting == "functional_impairment"
    explored = adapter.handle_message(
        sid,
        "这些低落和睡不好已经明显影响了工作。",
        expected_input="functional_impairment",
    )
    assert explored.l2_turn and explored.l2_turn.triggered_scales == ["PHQ-9"]
    assert explored.awaiting == "scale_consent"

    consent = adapter.handle_message(sid, "我同意。", expected_input="scale_consent")
    assert consent.l2_turn and consent.l2_turn.pending_scale_item is not None

    answers = [3, 3, 2, 2, 0, 2, 0, 0, 0]
    for score in answers:
        outcome = adapter.handle_message(
            sid, f"{score}分", expected_input="scale_answer"
        )
        assert outcome.accepted

    mania = adapter.handle_message(
        sid, "从未有过少睡却不困的时期。",
        expected_input="mania_or_hypomania_history",
    )
    assert mania.awaiting == "substance_or_medical_cause"
    substance = adapter.handle_message(
        sid, "这些变化并非药物、物质或身体疾病导致。",
        expected_input="substance_or_medical_cause",
    )
    assert substance.awaiting == "psychotic_symptoms"
    psychosis = adapter.handle_message(
        sid, "最近没有听到别人听不到的声音，也没有感觉被监视。",
        expected_input="psychotic_symptoms",
    )
    assert psychosis.l2_turn and psychosis.l2_turn.stage_after == "evoking"
    adapter.handle_message(
        sid, "我的目标是恢复工作状态和改善睡眠。", expected_input="patient_goals"
    )
    planning = adapter.handle_message(
        sid, "重要性8分，信心6分，准备度7分。", expected_input="motivation"
    )
    assert planning.l2_turn and planning.l2_turn.stage_after == "planning"
    return service, adapter, model, sid


def test_full_mi_flow_uses_two_model_calls_and_requires_confirmation_before_l3():
    service, adapter, model, sid = build_ready_session()
    intake = service.build_clinical_intake(sid)

    assert intake.information_sufficiency == "sufficient"
    assert intake.patient_confirmation == "pending"
    assert intake.scale_results["PHQ-9"].score == 12
    assert len(intake.scale_results["PHQ-9"].items) == 9
    assert all(item.sources for item in intake.key_symptoms_extracted)
    with pytest.raises(L2ExitGuardError):
        L2ToL3Adapter().assert_valid(intake)

    confirmed = adapter.handle_message(
        sid, "这些信息基本准确，继续。", expected_input="profile_confirmation"
    )
    assert confirmed.confirmation_decision == "continue"
    intake = service.build_clinical_intake(sid)
    receipt = L2ToL3Adapter().assert_valid(intake)
    profile = L2ToL3Adapter().to_profile(intake)

    assert receipt["route"] == "allow_l3"
    assert profile["PHQ9"] == 12
    assert profile["PHQ9_items"]["item1"] == 3
    assert model.analyze_calls == model.generate_calls
    assert model.analyze_calls >= 15


def test_scale_is_not_offered_in_first_reply_even_when_model_suggests_it():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("interview-before-scale"), session_id="interview-before-scale"
    )

    first = service.handle_message(
        sid, "最近1个月我睡不好，已经影响了工作，想先聊睡眠。"
    )
    assert first.accepted
    assert first.l2_turn is not None
    assert first.l2_turn.triggered_scales == []
    assert first.awaiting != "scale_consent"
    assert "量表" not in first.reply

    second = service.handle_message(
        sid, "睡不着时常常需要一个多小时才能入睡。",
        expected_input=first.awaiting or "auto",
    )
    assert second.accepted
    assert second.l2_turn is not None
    assert second.l2_turn.triggered_scales == ["ISI"]
    assert second.awaiting == "scale_consent"

    declined = service.handle_message(
        sid, "我拒绝。", expected_input="scale_consent"
    )
    assert declined.accepted
    assert service.get_session_snapshot(sid).scale_results["ISI"].status == "declined"


def test_repeated_extraction_of_same_focus_does_not_delay_scale_forever():
    class RepeatedFocusModel(ControlledL2Model):
        def analyze(self, patient_text, context):
            analysis = super().analyze(patient_text, context)
            if "睡" in patient_text:
                return analysis.model_copy(update={
                    "focus_topics": [
                        SemanticCandidate(value="睡眠", evidence_quote=patient_text)
                    ]
                })
            return analysis

    service = L2AssessmentService(
        model=RepeatedFocusModel(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("repeated-focus"), session_id="repeated-focus"
    )

    first = service.handle_message(
        sid, "最近1个月我睡不好，已经影响了工作，想先聊睡眠。"
    )
    assert first.l2_turn is not None
    assert first.l2_turn.triggered_scales == []
    second = service.handle_message(
        sid, "睡不着时常常需要一个多小时才能入睡。",
        expected_input=first.awaiting or "auto",
    )
    assert second.l2_turn is not None
    assert second.l2_turn.triggered_scales == ["ISI"]


def test_unsolicited_scale_invitation_is_rejected_before_interview_is_ready():
    class PrematureScaleModel(ControlledL2Model):
        def generate(self, decision: StageDecision, context):
            self.generate_calls += 1
            return "我们现在先做一份 ISI 量表，你愿意吗？"

    service = L2AssessmentService(
        model=PrematureScaleModel(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("premature-scale"), session_id="premature-scale"
    )
    before = service.get_session_snapshot(sid)

    result = service.handle_message(
        sid, "最近1个月我睡不好，已经影响了工作，想先聊睡眠。"
    )

    assert result.accepted is False
    assert result.error_code == "L2_RESPONSE_VALIDATION_FAILED"
    assert service.get_session_snapshot(sid) == before


def test_therapy_features_are_evidence_bound_and_forwarded_to_l3():
    patient_text = (
        "失眠是我的主要困扰，已经四个月，而且我会做重复仪式。"
        "这已经影响工作；我从未少睡却不困，也不是药物或身体疾病导致，"
        "最近没有听到别人听不到的声音。"
    )
    state = L2SessionState(
        session_id="therapy-features",
        turn_index=1,
        safety_passport=safe_passport("therapy-features"),
    )
    analysis = SemanticAnalysis(
        input_type="free_narrative",
        structured_fields=[
            StructuredFieldCandidate(
                field="insomnia_chief_complaint",
                value=True,
                evidence_quote="失眠是我的主要困扰",
            ),
            StructuredFieldCandidate(
                field="chronic_months",
                value=4,
                evidence_quote="已经四个月",
            ),
            StructuredFieldCandidate(
                field="ritual_behaviors",
                value=True,
                evidence_quote="我会做重复仪式",
            ),
            StructuredFieldCandidate(
                field="symptom_duration_days", value=120,
                evidence_quote="已经四个月",
            ),
            StructuredFieldCandidate(
                field="functional_impairment", value=True,
                evidence_quote="影响工作",
            ),
            StructuredFieldCandidate(
                field="mania_or_hypomania_history", value=False,
                evidence_quote="我从未少睡却不困",
            ),
            StructuredFieldCandidate(
                field="substance_or_medical_cause", value=False,
                evidence_quote="不是药物或身体疾病导致",
            ),
            StructuredFieldCandidate(
                field="psychotic_symptoms", value=False,
                evidence_quote="没有听到别人听不到的声音",
            ),
        ],
    )
    ProfileIngestor().ingest(
        state,
        analysis=analysis,
        patient_text=patient_text,
        message_id="therapy-message",
    )
    intake = ClinicalIntake(
        session_id=state.session_id,
        current_stage="planning",
        motivation=state.motivation,
        diagnostic_context=state.diagnostic_context,
        evidence_nodes=state.evidence_nodes,
        information_sufficiency="sufficient",
        patient_confirmation="confirmed",
    )

    profile = L2ToL3Adapter().to_profile(intake)

    assert profile["insomnia_chief_complaint"] is True
    assert profile["chronic_months"] == 4
    assert profile["ritual_behaviors"] is True


def test_risk2_commands_are_verified_and_persist_until_explicit_clear():
    model = ControlledL2Model()
    service = L2AssessmentService(model=model, require_fresh_passport=False)
    warned = safe_passport(
        "persistent-control",
        risk_level=2,
        safety_flag="warn",
        flags=["avoid_trauma_probing"],
        structured_labels={"scale_skip_list": ["PCL-5"]},
        execution={
            "route": "l2",
            "terminal": False,
            "applied_controls": ["avoid_trauma_probing", "skip_scale:PCL-5"],
        },
    )
    sid = service.create_session(warned, session_id="persistent-control")
    service.update_safety_passport(sid, safe_passport("persistent-control"))

    state = service.get_session_snapshot(sid)
    assert "avoid_trauma_probing" in state.active_safety_controls
    assert "skip_scale:PCL-5" in state.active_safety_controls
    assert state.scale_results["PCL-5"].status == "skipped"

    cleared = safe_passport(
        "persistent-control",
        execution={
            "route": "l2",
            "terminal": False,
            "applied_controls": [],
            "clear_controls": ["avoid_trauma_probing", "skip_scale:PCL-5"],
        },
    )
    service.update_safety_passport(sid, cleared)
    state = service.get_session_snapshot(sid)
    assert "avoid_trauma_probing" not in state.active_safety_controls
    assert "skip_scale:PCL-5" not in state.active_safety_controls
    assert state.scale_results["PCL-5"].status == "not_started"


def test_incomplete_safety_assessment_cannot_be_delegated_to_l2():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    passport = safe_passport(
        "one-shot-followup",
        risk_level=2,
        safety_flag="warn",
        risk_types=["suicidal_ideation"],
        decision_path="safety_assessment",
        flags=["monitor_suicidal_ideation"],
        structured_labels={
            "asq": {"status": "in_progress"},
            "l2_handoff": {
                "ordinary_care_allowed": False,
                "baseline_asq_status": "positive",
            },
        },
        execution={
            "route": "l1",
            "terminal": False,
            "commands": ["monitor_suicidal_ideation"],
            "applied_controls": ["monitor_suicidal_ideation"],
        },
        guard={"status": "pass", "can_call_l2": False},
    )

    with pytest.raises(L2EntranceGuardError):
        service.create_session(passport, session_id="one-shot-followup")


def test_guard_rejects_risk2_when_l1_command_has_no_execution_receipt():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    bad = safe_passport(
        "bad-control",
        risk_level=2,
        safety_flag="warn",
        flags=["avoid_trauma_probing"],
    )
    with pytest.raises(L2EntranceGuardError):
        service.create_session(bad, session_id="bad-control")


def test_guard_rejects_wrong_session_and_unsupported_control():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    wrong_session = safe_passport("another-session")
    with pytest.raises(L2EntranceGuardError):
        service.create_session(wrong_session, session_id="expected-session")

    unsupported = safe_passport(
        "unsupported-control",
        execution={
            "route": "l2",
            "terminal": False,
            "applied_controls": ["unknown_control"],
        },
    )
    with pytest.raises(L2EntranceGuardError):
        service.create_session(unsupported, session_id="unsupported-control")


def test_each_patient_message_must_match_a_fresh_l1_message_binding():
    service = L2AssessmentService(model=ControlledL2Model())
    sid = service.create_session(
        safe_passport("fresh-binding"), session_id="fresh-binding"
    )

    wrong = service.handle_message(sid, "最近压力有点大。", message_id="wrong")
    assert wrong.accepted is False
    assert wrong.error_code == "L2_SAFETY_MESSAGE_BINDING_INVALID"

    accepted = service.handle_message(
        sid, "最近压力有点大。", message_id="test-message"
    )
    assert accepted.accepted is True
    replayed = service.handle_message(
        sid, "换一条没有经过L1的新消息。", message_id="test-message"
    )
    assert replayed.accepted is False
    assert replayed.error_code == "L2_SAFETY_MESSAGE_BINDING_INVALID"


def test_new_l1_skip_command_stops_an_active_scale_immediately():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("active-scale-stop"), session_id="active-scale-stop"
    )
    adapter = L2DialogueAdapter(service)
    adapter.handle_message(
        sid, "最近三周我常做噩梦、容易惊吓，也焦虑和睡不好。"
    )
    adapter.handle_message(sid, "我想先聊那件事之后的噩梦和惊吓。")
    explored = adapter.handle_message(
        sid, "这些噩梦和惊吓已经明显影响了工作。",
        expected_input="functional_impairment",
    )
    assert explored.awaiting == "scale_consent"
    adapter.handle_message(sid, "我同意。", expected_input="scale_consent")
    assert service.get_session_snapshot(sid).scale_results["PCL-5"].status == "active"

    warned = safe_passport(
        "active-scale-stop",
        risk_level=2,
        safety_flag="warn",
        flags=["avoid_trauma_probing"],
        structured_labels={"scale_skip_list": ["PCL-5"]},
        execution={
            "route": "l2",
            "terminal": False,
            "applied_controls": ["avoid_trauma_probing", "skip_scale:PCL-5"],
        },
    )
    service.update_safety_passport(sid, warned)
    state = service.get_session_snapshot(sid)

    assert state.scale_results["PCL-5"].status == "skipped"
    assert state.pending_scale_name is None


def test_hallucinated_quote_is_not_written_to_profile():
    class HallucinatingModel(ControlledL2Model):
        def analyze(self, patient_text, context):
            self.analyze_calls += 1
            return SemanticAnalysis(
                input_type="free_narrative",
                symptoms=[SemanticCandidate(
                    value="depressed_mood", evidence_quote="患者没有说过的话"
                )],
            )

    service = L2AssessmentService(
        model=HallucinatingModel(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("evidence-check"), session_id="evidence-check"
    )
    result = service.handle_message(sid, "最近只是工作有点忙。")
    intake = service.build_clinical_intake(sid)

    assert result.accepted
    assert intake.key_symptoms_extracted == []
    assert intake.evidence_nodes == []


def test_model_failure_does_not_commit_partial_state():
    class FailingModel(ControlledL2Model):
        def analyze(self, patient_text, context):
            raise L2ModelUnavailable("timeout")

    service = L2AssessmentService(
        model=FailingModel(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("api-failure"), session_id="api-failure"
    )
    before = service.get_session_snapshot(sid)
    result = service.handle_message(sid, "最近很低落。")
    after = service.get_session_snapshot(sid)

    assert result.accepted is False
    assert result.error_code == "L2_MODEL_UNAVAILABLE"
    assert after == before


def test_invalid_scale_answer_is_not_committed():
    # 另建一条刚进入量表的会话，避免使用已经完成的 ready session。
    fresh = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    fresh_adapter = L2DialogueAdapter(fresh)
    fresh_sid = fresh.create_session(
        safe_passport("bad-score"), session_id="bad-score"
    )
    fresh_adapter.handle_message(
        fresh_sid,
        "最近三周我很低落、没兴趣、睡不好、没有精力、觉得自己让人失望。",
    )
    fresh_adapter.handle_message(fresh_sid, "我想先聊情绪低落。")
    explored = fresh_adapter.handle_message(
        fresh_sid,
        "这种低落和睡不好已经明显影响了工作。",
        expected_input="functional_impairment",
    )
    assert explored.awaiting == "scale_consent"
    fresh_adapter.handle_message(
        fresh_sid, "我同意。", expected_input="scale_consent"
    )
    before = fresh.get_session_snapshot(fresh_sid)
    result = fresh_adapter.handle_message(
        fresh_sid, "我不太确定。", expected_input="scale_answer"
    )
    after = fresh.get_session_snapshot(fresh_sid)

    assert result.accepted is False
    assert result.error_code == "L2_SCALE_ANSWER_AMBIGUOUS"
    assert after == before


def test_patient_visible_reply_never_exposes_internal_stage_name():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("smooth-transition"), session_id="smooth-transition"
    )
    result = service.handle_message(sid, "最近三周我情绪低落，也睡不好。")

    assert result.l2_turn and result.l2_turn.stage_after == "focusing"
    assert "focusing" not in result.reply.lower()
    assert "聚焦阶段" not in result.reply
    assert "进入下一阶段" not in result.reply


def test_exit_guard_rejects_evidence_bound_to_wrong_profile_field():
    service, adapter, _, sid = build_ready_session()
    adapter.handle_message(
        sid, "这些信息基本准确，继续。", expected_input="profile_confirmation"
    )
    intake = service.build_clinical_intake(sid).model_copy(deep=True)
    symptom = intake.key_symptoms_extracted[0]
    unrelated = next(
        node for node in intake.evidence_nodes
        if not node.target_path.startswith(f"symptoms.{symptom.symptom}")
    )
    symptom.sources = [unrelated.evidence_id]
    symptom.patient_quotes = [unrelated.quote]

    with pytest.raises(L2ExitGuardError, match="指向其他画像字段"):
        L2ToL3Adapter().assert_valid(intake)


@pytest.mark.parametrize("field", [
    "emotions", "cognitive_patterns", "patient_goals", "key_narratives",
    "functional_impairments", "focus_topics", "change_talk", "sustain_talk",
])
def test_fact_evidence_tracks_actual_index_for_new_and_merged_values(field):
    state = L2SessionState(
        session_id=f"evidence-index-{field}",
        safety_passport=safe_passport(f"evidence-index-{field}"),
    )
    state.turn_index = 1
    ingestor = ProfileIngestor()
    ingestor.ingest(
        state,
        analysis=SemanticAnalysis(input_type="free_narrative", **{
            field: [
                SemanticCandidate(value="甲", evidence_quote="甲一"),
                SemanticCandidate(value="乙", evidence_quote="乙一"),
                SemanticCandidate(value="甲", evidence_quote="甲二"),
            ],
        }),
        patient_text="甲一 乙一 甲二",
        message_id="first",
    )
    state.turn_index = 2
    ingestor.ingest(
        state,
        analysis=SemanticAnalysis(input_type="free_narrative", **{
            field: [SemanticCandidate(value="甲", evidence_quote="甲三")],
        }),
        patient_text="甲三",
        message_id="second",
    )

    facts = getattr(state, field)
    evidence = {node.evidence_id: node for node in state.evidence_nodes}
    assert [fact.value for fact in facts] == ["甲", "乙"]
    assert [evidence[source].target_path for source in facts[0].sources] == [
        f"{field}.0", f"{field}.0", f"{field}.0",
    ]
    assert [evidence[source].target_path for source in facts[1].sources] == [
        f"{field}.1",
    ]


@pytest.mark.parametrize("wrong_path", ["key_narratives.3", "key_narratives.1"])
def test_exit_guard_rejects_fact_evidence_at_wrong_index(wrong_path):
    service, adapter, _, sid = build_ready_session()
    adapter.handle_message(
        sid, "这些信息基本准确，继续。", expected_input="profile_confirmation"
    )
    intake = service.build_clinical_intake(sid).model_copy(deep=True)
    assert len(intake.key_narratives) >= 2
    source = intake.key_narratives[0].sources[0]
    node = next(item for item in intake.evidence_nodes if item.evidence_id == source)
    assert node.target_path == "key_narratives.0"
    node.target_path = wrong_path

    with pytest.raises(L2ExitGuardError, match="key_narratives 的证据指向其他画像字段"):
        L2ToL3Adapter().assert_valid(intake)


def test_yes_no_scale_answer_does_not_treat_not_as_yes():
    mdq_item = SCALE_CATALOG["MDQ"].items[0]
    analysis = SemanticAnalysis(input_type="scale_answer")

    assert ScaleEngine._extract_score("不是", analysis, mdq_item) == 0
    assert ScaleEngine._extract_score("是的", analysis, mdq_item) == 1


def _service_waiting_for_phq_item(item_number: int):
    class RecordingModel(ControlledL2Model):
        def __init__(self):
            super().__init__()
            self.last_decision = None
            self.last_context = None

        def generate(self, decision, context):
            self.last_decision = decision
            self.last_context = context
            return super().generate(decision, context)

    model = RecordingModel()
    service = L2AssessmentService(model=model, require_fresh_passport=False)
    passport = safe_passport("scale-response-context")
    sid = service.create_session(
        passport,
        session_id="scale-response-context",
    )
    state = service.get_session_snapshot(sid)
    state.current_stage = "focusing"
    state.pending_scale_name = "PHQ-9"
    state.scale_results["PHQ-9"] = ScaleResult(
        scale_name="PHQ-9",
        status="active",
        items={number: 0 for number in range(1, item_number)},
    )
    service.store.replace(state)
    return service, model, sid


def test_scale_reply_context_separates_answered_item_from_next_question():
    service, model, sid = _service_waiting_for_phq_item(1)

    result = service.handle_message(sid, "2", expected_input="scale_answer")

    assert result.accepted is True
    assert model.last_context["answered_scale_item"]["item_number"] == 1
    assert model.last_context["answered_scale_score"] == 2
    assert "情绪低落" in model.last_decision.required_question
    assert "提不起兴趣" not in model.last_decision.required_question


def test_response_model_receives_last_three_completed_dialogue_rounds():
    class RecordingModel(ControlledL2Model):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def generate(self, decision, context):
            self.contexts.append(context)
            return super().generate(decision, context)

    model = RecordingModel()
    service = L2AssessmentService(model=model, require_fresh_passport=False)
    sid = service.create_session(safe_passport("recent-dialogue"), session_id="recent-dialogue")
    patient_messages = [
        "最近工作压力很大。",
        "我主要想先聊睡眠。",
        "入睡通常要一个多小时。",
        "白天工作时也很疲惫。",
    ]

    replies = []
    for message in patient_messages:
        result = service.handle_message(sid, message)
        assert result.accepted is True
        replies.append(result.reply)

    assert model.contexts[0]["recent_dialogue"] == []
    assert model.contexts[-1]["recent_dialogue"] == [
        {"role": "patient", "text": patient_messages[0]},
        {"role": "assistant", "text": replies[0]},
        {"role": "patient", "text": patient_messages[1]},
        {"role": "assistant", "text": replies[1]},
        {"role": "patient", "text": patient_messages[2]},
        {"role": "assistant", "text": replies[2]},
    ]


def test_l2_records_phq9_item9_but_does_not_duplicate_l1_safety_protocol():
    service, model, sid = _service_waiting_for_phq_item(9)

    result = service.handle_message(sid, "1", expected_input="scale_answer")

    assert result.accepted is True
    assert result.awaiting == "focus_selection"
    assert model.last_context["answered_scale_item"]["item_number"] == 9
    assert model.last_context["answered_scale_score"] == 1
    assert "伤害自己或结束生命" not in result.reply


def test_isi_and_ybocs_keep_item_specific_response_anchors():
    isi = SCALE_CATALOG["ISI"]
    ybocs = SCALE_CATALOG["Y-BOCS"]

    assert isi.items[0].anchors[0] == "没有"
    assert isi.items[3].anchors[0] == "非常满意"
    assert isi.items[4].anchors[0] == "完全没有"
    assert ybocs.items[0].anchors[1] == "每天少于1小时"
    assert "努力抵抗" in ybocs.items[3].anchors[0]
    assert "完全能够控制" in ybocs.items[4].anchors[0]


def test_trauma_probe_reply_is_rejected_by_hard_rule_without_state_commit():
    class UnsafeReplyModel(ControlledL2Model):
        def generate(self, decision: StageDecision, context):
            self.generate_calls += 1
            return "请详细描述创伤，并回忆事件细节。"

    passport = safe_passport(
        "trauma-reply",
        risk_level=2,
        safety_flag="warn",
        flags=["avoid_trauma_probing"],
        execution={
            "route": "l2",
            "terminal": False,
            "applied_controls": ["avoid_trauma_probing"],
        },
    )
    service = L2AssessmentService(
        model=UnsafeReplyModel(), require_fresh_passport=False
    )
    sid = service.create_session(passport, session_id="trauma-reply")
    before = service.get_session_snapshot(sid)
    result = service.handle_message(sid, "最近工作压力很大。")

    assert result.accepted is False
    assert result.error_code == "L2_RESPONSE_VALIDATION_FAILED"
    assert service.get_session_snapshot(sid) == before


def test_patient_revision_replaces_old_goal_and_keeps_correction_trace():
    service, adapter, _, sid = build_ready_session()
    before = service.build_clinical_intake(sid)
    old_goal_sources = set(before.patient_goal_facts[0].sources)
    service.confirm_profile(sid, "revise")

    revised = adapter.handle_message(
        sid,
        "我的目标不是恢复工作状态，而是先恢复稳定睡眠。",
        expected_input="free_text",
    )
    intake = service.build_clinical_intake(sid)

    assert revised.accepted is True
    assert intake.patient_confirmation == "pending"
    assert intake.patient_goals == ["我的目标不是恢复工作状态，而是先恢复稳定睡眠。"]
    correction = next(
        node for node in intake.evidence_nodes
        if node.target_path == "corrections.patient_goals"
    )
    assert old_goal_sources.issubset(set(correction.parent_ids))
    evidence_by_id = {node.evidence_id: node for node in intake.evidence_nodes}
    assert all(
        evidence_by_id[source].target_path == "patient_goals.0"
        for source in intake.patient_goal_facts[0].sources
    )
    assert old_goal_sources.isdisjoint(intake.patient_goal_facts[0].sources)
    adapter.handle_message(
        sid, "这些信息基本准确，继续。", expected_input="profile_confirmation"
    )
    assert L2ToL3Adapter().assert_valid(service.build_clinical_intake(sid))["status"] == "pass"


def test_patient_can_pause_during_engaging_without_advancing_state():
    service = L2AssessmentService(
        model=ControlledL2Model(), require_fresh_passport=False
    )
    sid = service.create_session(
        safe_passport("pause-early"), session_id="pause-early"
    )
    result = service.handle_message(sid, "我想先暂停一下。")
    state = service.get_session_snapshot(sid)

    assert result.confirmation_decision == "pause"
    assert state.confirmation_status == "paused"
    assert state.current_stage == "engaging"
