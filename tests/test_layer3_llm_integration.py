import json

import pytest

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.llm import client
from tests.tree_test_support import node_payload


def test_production_clinicr_uses_api_for_all_three_steps(monkeypatch):
    monkeypatch.delenv("L3_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("L3_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen3.8-flash")
    profile = {
        "PHQ9": 16,
        "MDQ": 3,
        "PHQ9_items": {
            "item1": 3,
            "item2": 3,
            "item3": 2,
            "item4": 2,
            "item5": 2,
        },
    }
    mdd = {
        "disorder": "D013",
        "criteria_detail": [
            {
                "code": f"A{index}",
                "name": f"MDD条目{index}",
                "status": "met",
                "rationale": "单题证据支持",
                "evidence_refs": [f"PHQ9_item{index}={profile['PHQ9_items'][f'item{index}']}"],
            }
            for index in range(1, 6)
        ],
        "conclusion": "meets_criteria",
        "conclusion_reason": "满足5/9条并包含核心条目",
        "missing_dimensions": [],
    }
    gad = {
        "disorder": "D011",
        "criteria_detail": [{
            "code": "C1",
            "name": "过度担忧",
            "status": "pending",
            "rationale": "信息不足",
            "evidence_refs": [],
        }],
        "conclusion": "watch",
        "conclusion_reason": "部分满足，继续观察",
        "missing_dimensions": [{
            "field": "gad.duration",
            "purpose": "确认持续时间",
            "suggested_question": "这种担忧持续多久了？",
        }],
    }
    bipolar = {
        "disorder": "D009",
        "criteria_detail": [{
            "code": "CORE",
            "name": "轻躁狂史",
            "status": "not_met",
            "rationale": "MDQ筛查不支持",
            "evidence_refs": ["MDQ=3"],
        }],
        "conclusion": "ruled_out",
        "conclusion_reason": "不满足核心条件",
        "missing_dimensions": [],
    }
    verifier = {
        "score": 0.87,
        "dimensions": [
            {
                "dimension": dimension,
                "score": 0.87,
                "reason": "推理质量通过要求",
                "evidence_refs": ["PHQ9=16"] if dimension == "evidence_sufficiency" else [],
            }
            for dimension in (
                "evidence_sufficiency",
                "logic_coherence",
                "differential_coverage",
                "no_reasoning_leaps",
            )
        ],
        "fail_reasons": [],
        "missing_dimensions": [],
    }
    candidate_results = {"D013": mdd, "D011": gad, "D009": bipolar}
    calls = []

    def controlled_api(
        prompt,
        model="deepseek-chat",
        temperature=0,
        *,
        timeout=10,
        extra_body=None,
        response_format=None,
    ):
        calls.append({
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "timeout": timeout,
            "extra_body": extra_body,
            "response_format": response_format,
        })
        if "DSM-5 决策树节点判定器" in prompt:
            response = node_payload(prompt, ["D013", "D011", "D009"], "PHQ9_item2=3")
        elif "Reward Model Verifier" in prompt:
            response = verifier
        else:
            response = next(
                value
                for disorder, value in candidate_results.items()
                if f'"disorder":"{disorder}"' in prompt
            )
        return json.dumps(response, ensure_ascii=False)

    monkeypatch.setattr("layers.layer3.llm.client.ask_llm_stream", controlled_api)

    result = CLINICR().run(profile)

    assert result["diagnosis"]["primary"]["disorder"] == "D013"
    assert result["verifier_score"] == 0.87
    assert len(calls) > 5
    assert len([call for call in calls if "ELIMINATIVE" in call["prompt"]]) == 3
    assert result["candidate_generation_method"] == "dsm5_decision_tree"
    assert result["tree_trace"]
    assert all(call["temperature"] == 0 for call in calls)
    assert all(call["timeout"] == 60.0 for call in calls)
    assert all(call["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    } for call in calls)
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert all(
        call["response_format"] == {"type": "json_object"}
        for call in calls[1:4]
    )
    assert calls[4]["response_format"] == {"type": "json_object"}
    assert all("允许的 evidence_refs（只能复制 ref" in call["prompt"] for call in calls)
    assert "PHQ9=16" in calls[0]["prompt"]
    assert "PHQ9_item2=3" in calls[0]["prompt"]
    assert "_allowed_evidence_refs" not in calls[0]["prompt"]
    assert "\"confidence\"" not in calls[-1]["prompt"]


def test_step1_maps_exact_scalar_field_alias_to_traceable_evidence_node(monkeypatch):
    profile = {
        "symptom_duration_days": 45,
        "evidence_nodes": [{
            "evidence_id": "ev-duration",
            "target_path": "diagnostic_context.symptom_duration_days",
            "quote": "这一个多月",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-duration"],
    }
    prompts = []

    def controlled_api(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"nodes": [{"node_id": "criterion.D013.depressive_episode/min_duration",
                          "status": "met", "rationale": "45天病程",
                          "evidence_refs": ["symptom_duration_days=45"],
                          "symptom_ids": [], "confidence": 0.7}]}, ensure_ascii=False)

    monkeypatch.setattr("layers.layer3.llm.client.ask_llm_stream", controlled_api)

    result = client.APICLINICRReasoner(model="deepseek-chat")._evaluate_tree_nodes(profile, [{
        "node_id": "criterion.D013.depressive_episode/min_duration", "stage": "criterion", "disorders": ["D013"],
        "symptom_pool": [], "question": "是否达到来源规定病程？",
    }])

    assert all(item.evidence_refs == ["ev-duration"] for item in result)
    assert '"ref": "ev-duration"' in prompts[0]
    assert '"target_path": "diagnostic_context.symptom_duration_days"' in prompts[0]
    assert '"patient_quote": "这一个多月"' in prompts[0]


def test_l3_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("L3_LLM_TIMEOUT_SECONDS", "90")
    assert client._l3_timeout_seconds() == 90.0


@pytest.mark.parametrize("value", ["invalid", "0", "301"])
def test_invalid_l3_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("L3_LLM_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="L3_LLM_TIMEOUT_SECONDS"):
        client._l3_timeout_seconds()


def test_non_qwen_model_does_not_receive_qwen_reasoning_options():
    assert client._l3_reasoning_extra("deepseek-chat") is None


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_qwen_reasoning_effort_can_be_configured(monkeypatch, effort):
    monkeypatch.setenv("L3_REASONING_EFFORT", effort)
    assert client._l3_reasoning_extra("qwen3.8-flash") == {
        "enable_thinking": True,
        "reasoning_effort": effort,
    }


def test_invalid_qwen_reasoning_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("L3_REASONING_EFFORT", "high")
    with pytest.raises(ValueError, match="L3_REASONING_EFFORT"):
        client._l3_reasoning_extra("qwen3.8-flash")


def _step2_candidate():
    return client.DiagnosticCandidate.model_validate({
        "disorder": "D013",
        "label": "重性抑郁障碍",
        "symptom_match": {"matched": 1, "total": 9},
        "core_evidence_summary": "情绪低落",
        "evidence_refs": ["ev-mood"],
        "confidence": 0.7,
    })


def _valid_step2_payload():
    return {
        "disorder": "D013",
        "criteria_detail": [{
            "code": "A1",
            "name": "情绪低落",
            "status": "met",
            "rationale": "患者原话支持",
            "evidence_refs": ["ev-mood"],
        }],
        "conclusion": "watch",
        "conclusion_reason": "仍需补充其他标准",
        "missing_dimensions": [{
            "field": "mdd.additional_symptoms",
            "purpose": "确认其他症状",
            "suggested_question": "还出现了哪些变化？",
        }],
    }


def test_step2_retries_once_when_first_json_is_malformed(monkeypatch):
    responses = iter([
        '{"disorder":"D013" "criteria_detail":[]}',
        json.dumps(_valid_step2_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    profile = {
        "evidence_nodes": [{
            "evidence_id": "ev-mood",
            "target_path": "symptoms.depressed_mood",
            "quote": "最近一直情绪低落",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"],
    }

    result = client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
        profile, _step2_candidate()
    )

    assert result.disorder == "D013"
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert "上一次回答不是" not in calls[0]["prompt"]
    assert "上一次回答不是" in calls[1]["prompt"]


def test_step2_fails_closed_after_one_invalid_json_retry(monkeypatch):
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return '{"disorder":"D013" "criteria_detail":[]}'

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    profile = {client.ALLOWED_EVIDENCE_REFS_KEY: []}

    with pytest.raises(client.CLINICRModelUnavailable, match="JSONDecodeError"):
        client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
            profile, _step2_candidate()
        )

    assert len(calls) == 2


def test_step2_retries_unknown_evidence_without_silently_accepting_it(monkeypatch):
    invalid = _valid_step2_payload()
    invalid["criteria_detail"][0]["evidence_refs"] = ["invented-evidence-id"]
    responses = iter([
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(_valid_step2_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    profile = {
        "evidence_nodes": [{
            "evidence_id": "ev-mood",
            "target_path": "symptoms.depressed_mood",
            "quote": "最近一直情绪低落",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"],
    }

    result = client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
        profile, _step2_candidate()
    )

    assert result.criteria_detail[0].evidence_refs == ["ev-mood"]
    assert len(calls) == 2
    assert "证据白名单外" in calls[1]


def test_step2_fails_closed_when_retry_still_invents_evidence(monkeypatch):
    invalid = _valid_step2_payload()
    invalid["criteria_detail"][0]["evidence_refs"] = ["invented-evidence-id"]
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(invalid, ensure_ascii=False)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    profile = {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]}

    with pytest.raises(client.CLINICRModelUnavailable, match="ValueError"):
        client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
            profile, _step2_candidate()
        )

    assert len(calls) == 2


def test_step2_retries_provider_structured_json_api_error_once(monkeypatch):
    class FakeAPIError(Exception):
        pass

    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise FakeAPIError(
                "Model output became abnormal while generating a JSON response "
                "for response_format; partial output may be incomplete or invalid JSON"
            )
        return json.dumps(_valid_step2_payload(), ensure_ascii=False)

    monkeypatch.setattr(client, "APIError", FakeAPIError)
    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)
    profile = {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]}

    result = client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
        profile, _step2_candidate()
    )

    assert result.disorder == "D013"
    assert len(calls) == 2


def test_step2_reports_non_structured_api_failure_as_api_error_without_retry(monkeypatch):
    class FakeAPIError(Exception):
        pass

    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        raise FakeAPIError("authentication failed")

    monkeypatch.setattr(client, "APIError", FakeAPIError)
    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    with pytest.raises(client.CLINICRAPIError, match="Step 2 API错误"):
        client.APICLINICRReasoner("deepseek-chat").evaluate_candidate(
            {client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"]},
            _step2_candidate(),
        )

    assert len(calls) == 1


def _valid_verifier_payload():
    return {
        "score": 0.72,
        "dimensions": [
            {
                "dimension": dimension,
                "score": 0.72,
                "reason": "推理过程基本完整",
                "evidence_refs": ["ev-mood"] if dimension == "evidence_sufficiency" else [],
            }
            for dimension in (
                "evidence_sufficiency",
                "logic_coherence",
                "differential_coverage",
                "no_reasoning_leaps",
            )
        ],
        "fail_reasons": [],
        "missing_dimensions": [{
            "field": "appetite_change",
            "purpose": "补充症状证据",
            "suggested_question": "最近食欲有什么变化？",
        }],
    }


def _score_with_test_inputs(model):
    profile = {
        "evidence_nodes": [{
            "evidence_id": "ev-mood",
            "target_path": "symptoms.depressed_mood",
            "quote": "最近一直情绪低落",
        }],
        client.ALLOWED_EVIDENCE_REFS_KEY: ["ev-mood"],
    }
    assessment = client.EliminativeAssessment.model_validate(_valid_step2_payload())
    return model.score_reasoning(
        profile,
        {"disorder": "D013", "evidence_refs": ["ev-mood"]},
        assessment,
        [{"disorder": "D011", "conclusion": "watch"}],
    )


def test_step3_retries_once_when_first_payload_uses_wrong_field(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["missing_dimensions"][0]["suggested_question_for_L2"] = (
        invalid["missing_dimensions"][0].pop("suggested_question")
    )
    responses = iter([
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(_valid_verifier_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier("deepseek-chat"))

    assert result.score == 0.72
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert "禁止使用 suggested_question_for_L2" not in calls[0]["prompt"]
    assert "禁止使用 suggested_question_for_L2" in calls[1]["prompt"]


def test_step3_fails_closed_after_one_invalid_schema_retry(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["missing_dimensions"][0]["suggested_question_for_L2"] = (
        invalid["missing_dimensions"][0].pop("suggested_question")
    )
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(invalid, ensure_ascii=False)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    with pytest.raises(client.CLINICRModelUnavailable, match="ValidationError"):
        _score_with_test_inputs(client.APIRewardModelVerifier("deepseek-chat"))

    assert len(calls) == 2


def test_step3_retries_unknown_evidence_and_keeps_whitelist_enforced(monkeypatch):
    invalid = _valid_verifier_payload()
    invalid["dimensions"][0]["evidence_refs"] = ["invented-evidence-id"]
    responses = iter([
        json.dumps(invalid, ensure_ascii=False),
        json.dumps(_valid_verifier_payload(), ensure_ascii=False),
    ])
    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier("deepseek-chat"))

    assert result.dimensions[0].evidence_refs == ["ev-mood"]
    assert len(calls) == 2
    assert "证据白名单外" in calls[1]


def test_step3_retries_provider_structured_json_api_error_once(monkeypatch):
    class FakeAPIError(Exception):
        pass

    calls = []

    def controlled_api(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise FakeAPIError(
                "JSON response for response_format was incomplete and invalid"
            )
        return json.dumps(_valid_verifier_payload(), ensure_ascii=False)

    monkeypatch.setattr(client, "APIError", FakeAPIError)
    monkeypatch.setattr(client, "ask_llm_stream", controlled_api)

    result = _score_with_test_inputs(client.APIRewardModelVerifier("deepseek-chat"))

    assert result.score == 0.72
    assert len(calls) == 2
