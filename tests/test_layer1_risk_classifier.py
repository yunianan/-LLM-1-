import json

import pytest

from layers.layer1 import detector, risk_classifier
from layers.layer1.risk_classifier import L1ModelFailure


VALID = {
    "suicide_signal_detected": False,
    "jailbreak_detected": False,
    "safe_to_enter_l2": True,
    "confidence": 0.9,
    "evidence_quotes": [],
    "jailbreak_evidence": [],
    "summary": "未发现风险",
}


def test_accepts_strict_json(monkeypatch):
    monkeypatch.setattr(risk_classifier, "ask_llm_stream", lambda _prompt, **_kwargs: json.dumps(VALID))
    result = risk_classifier.llm_classify("最近压力较大")
    assert result["analysis"]["suicide_signal_detected"] is False
    assert result["llm_score"] == 1
    assert result["attempts"] == 1


def test_accepts_markdown_json_fence(monkeypatch):
    response = "```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```"
    monkeypatch.setattr(risk_classifier, "ask_llm_stream", lambda _prompt, **_kwargs: response)
    assert risk_classifier.llm_classify("最近压力较大")["analysis"]["safe_to_enter_l2"] is True


def test_retries_once_after_invalid_json(monkeypatch):
    responses = iter(["{incomplete", json.dumps(VALID)])
    monkeypatch.setattr(risk_classifier, "ask_llm_stream", lambda _prompt, **_kwargs: next(responses))
    assert risk_classifier.llm_classify("最近压力较大")["attempts"] == 2


def test_invalid_schema_is_distinguished_from_api_error(monkeypatch):
    monkeypatch.setattr(risk_classifier, "ask_llm_stream", lambda _prompt, **_kwargs: json.dumps({"confidence": 2}))
    with pytest.raises(L1ModelFailure) as exc:
        risk_classifier.llm_classify("最近压力较大")
    assert exc.value.code == "L1_MODEL_OUTPUT_INVALID"
    assert exc.value.stage == "L1.signal.schema_validation"


def test_api_timeout_has_distinct_code(monkeypatch):
    monkeypatch.setattr(
        risk_classifier,
        "ask_llm_stream",
        lambda _prompt, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    with pytest.raises(L1ModelFailure) as exc:
        risk_classifier.llm_classify("最近压力较大")
    assert exc.value.code == "L1_API_ERROR"
    assert exc.value.stage == "L1.signal.api_request"


def test_detector_converts_api_error_to_fail_closed_result(monkeypatch):
    def failed(*_args, **_kwargs):
        raise L1ModelFailure(
            code="L1_API_ERROR",
            stage="L1.signal.api_request",
            exception_type="APITimeoutError",
        )

    monkeypatch.setattr(detector, "llm_classify", failed)
    result = detector.safety_detect("最近压力较大", session_id="api", message_id="m")
    assert result["error_code"] == "L1_API_ERROR"
    assert result["terminal"] is True
    assert result["can_call_l2"] is False


def test_l1_uses_streaming_with_sixty_second_timeout(monkeypatch):
    calls = []

    def streaming(_prompt, **kwargs):
        calls.append(kwargs)
        return json.dumps(VALID)

    monkeypatch.delenv("L1_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(risk_classifier, "ask_llm_stream", streaming)
    risk_classifier.llm_classify("最近压力较大")
    assert calls == [{"timeout": 60.0, "response_format": {"type": "json_object"}}]


def test_evidence_quote_must_come_from_current_message(monkeypatch):
    invalid = {
        **VALID,
        "suicide_signal_detected": True,
        "safe_to_enter_l2": False,
        "evidence_quotes": ["患者没有说过的文本"],
    }
    monkeypatch.setattr(
        risk_classifier,
        "ask_llm_stream",
        lambda _prompt, **_kwargs: json.dumps(invalid, ensure_ascii=False),
    )
    with pytest.raises(L1ModelFailure) as exc:
        risk_classifier.llm_classify("我最近压力很大")
    assert exc.value.code == "L1_MODEL_OUTPUT_INVALID"


def test_route_fields_must_be_consistent(monkeypatch):
    inconsistent = {
        **VALID,
        "suicide_signal_detected": True,
        "safe_to_enter_l2": True,
    }
    monkeypatch.setattr(
        risk_classifier,
        "ask_llm_stream",
        lambda _prompt, **_kwargs: json.dumps(inconsistent, ensure_ascii=False),
    )
    with pytest.raises(L1ModelFailure) as exc:
        risk_classifier.llm_classify("我不想活了")
    assert exc.value.code == "L1_MODEL_OUTPUT_INVALID"


def test_suicide_signal_requires_asq_transition_support(monkeypatch):
    missing_transition = {
        **VALID,
        "suicide_signal_detected": True,
        "safe_to_enter_l2": False,
        "evidence_quotes": ["我不想活了"],
    }
    monkeypatch.setattr(
        risk_classifier,
        "ask_llm_stream",
        lambda _prompt, **_kwargs: json.dumps(missing_transition, ensure_ascii=False),
    )
    with pytest.raises(L1ModelFailure) as exc:
        risk_classifier.llm_classify("我不想活了")
    assert exc.value.code == "L1_MODEL_OUTPUT_INVALID"


def test_failure_log_does_not_contain_patient_text_or_exception_body(monkeypatch, caplog):
    def unavailable(_prompt, **_kwargs):
        raise TimeoutError("sensitive diagnostic body")

    monkeypatch.setattr(risk_classifier, "ask_llm_stream", unavailable)
    with caplog.at_level("ERROR"), pytest.raises(L1ModelFailure):
        risk_classifier.llm_classify("患者原话不应进入日志")
    assert "L1_MODEL_FAILURE" in caplog.text
    assert "TimeoutError" in caplog.text
    assert "sensitive diagnostic body" not in caplog.text
    assert "患者原话不应进入日志" not in caplog.text
