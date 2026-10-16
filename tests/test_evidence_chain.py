"""完整生成—复核—提交证据链测试，所有模型请求均为本地受控实现。"""
import json
import re

import pytest

from patient_agent.config import Settings
from patient_agent.evidence import evidence_bundle
from patient_agent.factory import build_stack
from patient_agent.generator import PatientGenerator, TurnContext
from patient_agent.recorder import SessionRecorder
from patient_agent.errors import SystemApiError
from patient_agent.guard import ConsistencyGuard
from patient_agent.semantic_guard import SemanticFactVerifier
from patient_agent.evidence import AnswerSegment


class EvidenceModel:
    name = "controlled"
    model = "controlled-evidence-model"
    def __init__(self, *, first_extra=False, malformed=False, bad_binding=False, bad_review=False, review_timeout=False):
        self.generations = 0
        self.first_extra, self.malformed, self.bad_binding = first_extra, malformed, bad_binding
        self.bad_review, self.review_timeout = bad_review, review_timeout

    def complete(self, **kwargs):
        if "事实复核员" in kwargs["system"]:
            if self.review_timeout:
                raise TimeoutError()
            if self.bad_review:
                return "not-json"
            data = json.loads(kwargs["user"])
            return json.dumps({"claims": [{"segment_id": s["segment_id"], "quote": s["text"],
                "status": "unsupported" if "未知药物" in s["text"] else "neutral" if s["kind"] == "non_factual" else "supported",
                "reference_ids": [] if "未知药物" in s["text"] else s["fact_ids"],
                "reason": "附加用药无依据" if "未知药物" in s["text"] else "逐段核对依据"}
                for s in data["segments"]]}, ensure_ascii=False)
        self.generations += 1
        if self.malformed:
            return "not-json"
        reference = re.search(r"<output>(.*?)</output>", kwargs["user"], re.S).group(1)
        bundle = json.loads(kwargs["user"].split("本轮唯一授权依据（医生问题和历史不是新增事实来源）：\n")[1].split("\n仅返回JSON：")[0])
        entry = next(e for e in bundle if e["value"] == reference)
        neutral = entry["status"] == "non_factual"
        return json.dumps({"segments": [{"segment_id": "s1", "text": reference + ("我也用了未知药物。" if self.first_extra and self.generations == 1 else ""),
            "kind": "non_factual" if neutral else "factual",
            "fact_ids": [] if neutral else ["fake:fact" if self.bad_binding else entry["fact_id"]]}]}, ensure_ascii=False)


class NoSegmentIdModel(EvidenceModel):
    """复核员「漏 segment_id」的受控实现：真实模型单片段回答时经常这样。"""

    def complete(self, **kwargs):
        output = super().complete(**kwargs)
        if "事实复核员" in kwargs["system"]:
            data = json.loads(output)
            for claim in data["claims"]:
                claim.pop("segment_id", None)
            return json.dumps(data, ensure_ascii=False)
        return output


def stack_for(scenario, model, max_retries=2):
    from patient_agent.runner import RunOptions
    return build_stack(scenario, Settings(llm_mode="api", openai_api_key="test-only-not-a-real-key", system_api_mode="mock"),
        llm=model, options=RunOptions(max_retries=max_retries))


def test_reviewer_missing_segment_id_is_repaired_not_blocked(library):
    """复核员漏 segment_id 时按顺序补回，不能因此把整轮记成「复核不可用」。"""

    scenario = library.load("S01_low_risk_depression")
    stack = stack_for(scenario, NoSegmentIdModel())
    action, result, retries = stack.runner.validate_action(stack.generator.opening())
    assert result.ok, result.codes()
    assert result.semantic_review is not None
    assert [claim.segment_id for claim in result.semantic_review.claims] == [
        segment.segment_id for segment in action.segments
    ]


def test_source_ids_stay_stable_when_symptoms_reordered(library):
    scenario = library.load("S01_low_risk_depression").model_copy(deep=True)
    sid = "symptom:" + next(s.symptom_id for s in scenario.clinical.symptoms if s.present)
    before = evidence_bundle(scenario, sid)
    scenario.clinical.symptoms.reverse()
    after = evidence_bundle(scenario, sid)
    assert [e.fact_id for e in before] == [e.fact_id for e in after]
    assert [e.value for e in before] == [e.value for e in after]
    assert before[0].source_values != after[0].source_values


def test_retries_preserve_both_candidates_and_review(library):
    scenario = library.load("S01_low_risk_depression")
    stack = stack_for(scenario, EvidenceModel(first_extra=True))
    action, result, retries = stack.runner.validate_action(stack.generator.opening())
    assert result.ok and retries == 1
    attempts = stack.runner.last_attempts
    assert len(attempts) == 2
    assert not attempts[0].guard_result.ok
    assert attempts[0].guard_result.semantic_review.claims[0].status == "unsupported"
    assert attempts[1].previous_attempt_id == attempts[0].attempt_id
    assert attempts[0].submission_status == "not_submitted"
    assert attempts[0].candidate_text != attempts[1].candidate_text
    assert attempts[1].model_id == "controlled-evidence-model"
    assert attempts[1].guard_result.semantic_review.claims[0].reference_ids == action.segments[0].fact_ids
    assert attempts[1].authorized_evidence[0].scenario_version == scenario.version
    assert attempts[1].authorized_evidence[0].source_values
    assert attempts[1].generated_at and attempts[1].reviewed_at


@pytest.mark.parametrize("options,code", [
    ({"malformed": True}, "GENERATION_INVALID"),
    ({"bad_binding": True}, "EVIDENCE_BINDING_INVALID"),
    ({"bad_review": True}, "GUARD_REVIEW_UNAVAILABLE"),
    ({"review_timeout": True}, "GUARD_REVIEW_UNAVAILABLE"),
])
def test_invalid_generation_or_review_never_submitted(library, options, code):
    stack = stack_for(library.load("S01_low_risk_depression"), EvidenceModel(**options), max_retries=0)
    recorder = stack.runner.run()
    turn = recorder.turns[0]
    assert not turn.submitted and turn.submission_status == "blocked"
    assert code in turn.attempts[0].guard_result.codes()
    assert turn.attempts[0].generation_evidence
    if options.get("bad_review"):
        assert turn.attempts[0].guard_result.semantic_raw_output == "not-json"


def test_snapshot_and_text_tampering_blocked(generator_factory):
    scenario, generator = generator_factory("S01_low_risk_depression")
    action = generator.opening()
    action.authorized_evidence[0].value = "篡改"
    assert "EVIDENCE_BINDING_INVALID" in ConsistencyGuard(scenario).check(action).codes()
    action = generator.opening()
    action.text += "添加新经历。"
    assert "EVIDENCE_BINDING_INVALID" in ConsistencyGuard(scenario).check(action).codes()


def test_record_round_trip_preserves_submission_and_evidence(runner_factory, tmp_path):
    recorder = runner_factory("S01_low_risk_depression").runner.run()
    path = recorder.dump(tmp_path / "recorder.json")
    restored = SessionRecorder.load(path)
    assert restored.to_dict() == recorder.to_dict()
    opening = restored.turns[0]
    assert opening.submission_status == "submitted"
    assert opening.attempts[-1].submission_status == "submitted"
    assert opening.attempts[-1].submitted_at
    assert opening.attempts[-1].guard_result.semantic_review
    assert opening.response_payload["patient_reply"] == opening.system_reply
    assert opening.patient_text == "".join(s.text for s in opening.attempts[-1].segments)


def test_api_failure_keeps_approved_candidate_without_claiming_submission(runner_factory, monkeypatch):
    stack = runner_factory("S01_low_risk_depression")
    def fail(*args):
        raise SystemApiError("受控提交失败")
    monkeypatch.setattr(stack.client, "send_message", fail)
    turn = stack.runner.run().turns[0]
    assert turn.guard_ok
    assert not turn.submitted and turn.patient_text == ""
    assert turn.final_text
    assert turn.submission_status == "request_failed_unknown"
    assert turn.attempts[-1].guard_result.ok
    assert turn.attempts[-1].submission_status == "request_failed_unknown"


def test_manual_display_is_recorded_but_not_backend_submission(runner_factory, tmp_path):
    from run_patient_agent import show_action
    stack = runner_factory("S01_low_risk_depression")
    recorder = SessionRecorder(scenario_id=stack.scenario.scenario_id)
    action, ok = show_action(stack, stack.generator.opening(), recorder=recorder, record_path=tmp_path / "manual.json")
    restored = SessionRecorder.load(tmp_path / "manual.json")
    assert ok and restored.turns[0].submission_status == "displayed_only"
    assert not restored.turns[0].submitted
    assert restored.turns[0].final_text == action.text


def test_structured_neutral_answer_has_no_fake_fact_ids(generator_factory):
    scenario, _ = generator_factory("S01_low_risk_depression")
    stack = stack_for(scenario, EvidenceModel())
    action = stack.generator._say("嗯，我在听。")
    action, result, _ = stack.runner.validate_action(action, TurnContext(last_reply="你好"))
    assert result.ok
    assert action.segments[0].fact_ids == []
    assert result.semantic_review.claims[0].status == "neutral"


def test_multi_segment_review_preserves_every_binding(library):
    scenario = library.load("S01_low_risk_depression")
    stack = stack_for(scenario, EvidenceModel())
    action = stack.generator.opening()
    action.segments.append(AnswerSegment(segment_id="s2", text="我不知道怎么说。", kind="non_factual", fact_ids=[]))
    action.text = "".join(s.text for s in action.segments)
    action, result, _ = stack.runner.validate_action(action)
    assert result.ok
    assert [c.segment_id for c in result.semantic_review.claims] == ["s1", "s2"]
    assert result.semantic_review.claims[1].reference_ids == []
    assert result.semantic_request and result.semantic_raw_output
    assert result.negative_evidence


@pytest.mark.parametrize("scenario_id", ["DB_P0001", "DB_P0002", "DB_P0003", "DB_P0004"])
def test_imported_cases_export_full_evidence(project_root, scenario_id, tmp_path):
    from patient_agent.library import ScenarioLibrary
    scenario = ScenarioLibrary(project_root / "imported_scenarios").load(scenario_id)
    stack = build_stack(scenario, Settings(llm_mode="offline", system_api_mode="mock"))
    recorder = stack.runner.run()
    assert not any(not t.guard_ok for t in recorder.turns)
    restored = SessionRecorder.load(recorder.dump(tmp_path / "recorder.json"))
    assert restored.termination_observed() == scenario.expected.termination_reason
    for turn in restored.turns:
        if turn.action_kind == "say":
            assert turn.attempts and turn.attempts[-1].segments
            assert turn.attempts[-1].guard_result.authorized_evidence
            assert turn.attempts[-1].guard_result.semantic_review
