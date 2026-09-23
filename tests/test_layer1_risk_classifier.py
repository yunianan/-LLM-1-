import json

import pytest

from layers.layer1 import detector, risk_classifier


VALID = {
    "risk_level": 1,
    "score": 1,
    "risk_types": [],
    "confidence": 0.9,
    "reason": "未发现紧急风险",
}


def test_accepts_strict_json(monkeypatch):
    monkeypatch.setattr(risk_classifier, "ask_llm", lambda _prompt: json.dumps(VALID))

    result = risk_classifier.llm_classify("最近压力较大")

    assert result["llm_score"] == 1
    assert result["degraded"] is False
    assert result["attempts"] == 1


def test_accepts_markdown_json_fence(monkeypatch):
    response = "```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```"
    monkeypatch.setattr(risk_classifier, "ask_llm", lambda _prompt: response)

    result = risk_classifier.llm_classify("最近压力较大")

    assert result["degraded"] is False


def test_retries_once_after_invalid_json(monkeypatch):
    responses = iter(["{incomplete", json.dumps(VALID)])
    monkeypatch.setattr(risk_classifier, "ask_llm", lambda _prompt: next(responses))

    result = risk_classifier.llm_classify("最近压力较大")

    assert result["degraded"] is False
    assert result["attempts"] == 2


@pytest.mark.parametrize("bad", [
    {"risk_level": 1, "score": 1, "risk_types": [], "confidence": 0.9},
    {**VALID, "risk_level": "3"},
    {**VALID, "score": 4},
    {**VALID, "confidence": 1.5},
])
def test_invalid_schema_fails_closed(monkeypatch, bad):
    monkeypatch.setattr(risk_classifier, "ask_llm", lambda _prompt: json.dumps(bad))

    result = risk_classifier.llm_classify("最近压力较大")

    assert result["degraded"] is True
    assert result["llm_score"] == 2
    assert result["confidence"] == 0.0
    assert result["attempts"] == 2


def test_service_unavailable_returns_terminal_error(monkeypatch):
    def unavailable(_prompt):
        raise TimeoutError("timeout")

    monkeypatch.setattr(risk_classifier, "ask_llm", unavailable)

    result = detector.safety_detect("最近压力较大")

    assert result["status"] == "error"
    assert result["error_code"] == "L1_LLM_UNAVAILABLE"
    assert result["message"] == "安全评估模型不可用"
    assert result["terminal"] is True
    assert result["can_call_l2"] is False
    assert result["details"]["attempts"] == 2


def test_unknown_sdk_exception_also_fails_closed(monkeypatch):
    monkeypatch.setattr(
        risk_classifier, "ask_llm", lambda _prompt: (_ for _ in ()).throw(RuntimeError("sdk"))
    )

    result = risk_classifier.llm_classify("最近压力较大")

    assert result["degraded"] is True
    assert "RuntimeError" in result["failure_reason"]


def test_llm_failure_returns_error_even_if_rules_detect_crisis(monkeypatch):
    monkeypatch.setattr(
        risk_classifier, "ask_llm", lambda _prompt: (_ for _ in ()).throw(ConnectionError())
    )

    result = detector.safety_detect("我不想活了，想结束生命")

    assert result["status"] == "error"
    assert result["message"] == "安全评估模型不可用"
    assert result["can_call_l2"] is False
    assert result["details"]["rule_score"] == 3
    assert "suicidal_ideation" in result["details"]["rule_risk_types"]
