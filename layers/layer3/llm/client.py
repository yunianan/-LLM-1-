from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Protocol, TypeVar

from openai import APIError

from shared.llm_client import ask_llm_stream, get_model

from ..diagnosis.schemas import (
    CandidateGeneration,
    DiagnosticCandidate,
    EliminativeAssessment,
    RewardModelAssessment,
    SUPPORTED_DISORDERS,
)


ALLOWED_EVIDENCE_REFS_KEY = "_allowed_evidence_refs"
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}
StructuredResult = TypeVar("StructuredResult")


class CLINICRModelUnavailable(RuntimeError):
    pass


class CLINICRAPIError(CLINICRModelUnavailable):
    """CLINICR 模型服务/API 调用失败，与模型输出校验失败区分。"""


def _is_retryable_structured_output_api_error(error: APIError) -> bool:
    """只重试供应商明确标记的 JSON response_format 生成中止。"""
    message = str(error).casefold()
    return (
        "response_format" in message
        and "json" in message
        and any(marker in message for marker in ("invalid", "incomplete", "abnormal"))
    )


def _request_structured_output_with_one_retry(
    request: Callable[[str], str],
    validate: Callable[[str], StructuredResult],
    prompt: str,
    retry_prompt: str,
) -> StructuredResult:
    """解析失败或供应商中止 JSON 生成时，共享同一个单次重试额度。"""
    try:
        raw = request(prompt)
    except APIError as exc:
        if not _is_retryable_structured_output_api_error(exc):
            raise
        return validate(request(retry_prompt))
    try:
        return validate(raw)
    except ValueError:
        return validate(request(retry_prompt))


def _l3_timeout_seconds() -> float:
    """L3 诊断输出较长，为三个 CLINICR 模型步骤提供独立超时。"""
    raw = (os.getenv("L3_LLM_TIMEOUT_SECONDS") or "60").strip()
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("L3_LLM_TIMEOUT_SECONDS 必须是数字") from exc
    if not 1 <= timeout <= 300:
        raise ValueError("L3_LLM_TIMEOUT_SECONDS 必须在 1 到 300 秒之间")
    return timeout


def _l3_reasoning_extra(model: str) -> dict[str, Any] | None:
    """Qwen3 保留思考模式，但避免使用耗时过长的默认 xhigh。"""
    if "qwen3" not in model.casefold():
        return None
    effort = (os.getenv("L3_REASONING_EFFORT") or "medium").strip().lower()
    if effort not in {"low", "medium", "xhigh"}:
        raise ValueError("L3_REASONING_EFFORT 必须是 low、medium 或 xhigh")
    return {"enable_thinking": True, "reasoning_effort": effort}


class CLINICRReasoner(Protocol):
    def generate_candidates(self, profile: dict[str, Any]) -> CandidateGeneration:
        ...

    def evaluate_candidate(
        self,
        profile: dict[str, Any],
        candidate: DiagnosticCandidate,
    ) -> EliminativeAssessment:
        ...


class RewardModelVerifier(Protocol):
    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        ...


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return value


def _prompt_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """从规则注入的元数据中拆出唯一允许引用列表，不把元数据当临床事实。"""
    raw_refs = profile.get(ALLOWED_EVIDENCE_REFS_KEY, [])
    allowed = (
        [item for item in raw_refs if isinstance(item, str) and item]
        if isinstance(raw_refs, list)
        else []
    )
    clinical_profile = {
        key: value
        for key, value in profile.items()
        if key != ALLOWED_EVIDENCE_REFS_KEY
    }
    return clinical_profile, allowed


def _evidence_catalog(
    clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> list[dict[str, str]]:
    """给模型提供可选择证据的含义，避免它根据画像字段另造引用。"""
    node_by_id = {
        str(node["evidence_id"]): node
        for node in clinical_profile.get("evidence_nodes") or []
        if isinstance(node, dict) and node.get("evidence_id")
    }
    catalog: list[dict[str, str]] = []
    for ref in allowed_refs:
        item = {"ref": ref}
        node = node_by_id.get(ref)
        if node:
            if node.get("target_path"):
                item["target_path"] = str(node["target_path"])
            quote = node.get("quote") or node.get("evidence_quote")
            if quote:
                item["patient_quote"] = str(quote)
        catalog.append(item)
    return catalog


def _value_at_evidence_target(
    clinical_profile: dict[str, Any], target_path: str
) -> Any:
    """读取证据节点所指向的画像值；兼容 L2→L3 的扁平诊断字段。"""
    missing = object()
    parts = target_path.split(".")
    current: Any = clinical_profile
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            current = missing
            break
        current = current[part]
    if current is not missing:
        return current
    if len(parts) == 2 and parts[0] == "diagnostic_context":
        return clinical_profile.get(parts[1], missing)
    return missing


def _canonical_evidence_aliases(
    clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> dict[str, list[str]]:
    """把模型常见的 ``字段=值`` 写法映射回可追溯证据节点。"""
    allowed = set(allowed_refs)
    aliases: dict[str, list[str]] = {}
    for node in clinical_profile.get("evidence_nodes") or []:
        if not isinstance(node, dict):
            continue
        evidence_id = str(node.get("evidence_id") or "")
        target_path = str(node.get("target_path") or "")
        if not evidence_id or evidence_id not in allowed or not target_path:
            continue
        value = _value_at_evidence_target(clinical_profile, target_path)
        if type(value) not in {str, int, float, bool}:
            continue
        serialized_values = {
            str(value),
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        }
        field = target_path.rsplit(".", 1)[-1]
        for serialized in serialized_values:
            for alias in (f"{field}={serialized}", f"{target_path}={serialized}"):
                aliases.setdefault(alias, []).append(evidence_id)
    return aliases


def _canonicalize_payload_evidence_refs(
    payload: dict[str, Any], clinical_profile: dict[str, Any], allowed_refs: list[str]
) -> dict[str, Any]:
    """仅将能由真实字段和值唯一解释的别名换成白名单证据 ID。"""
    allowed = set(allowed_refs)
    aliases = _canonical_evidence_aliases(clinical_profile, allowed_refs)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_refs" and isinstance(child, list):
                    normalized: list[Any] = []
                    for ref in child:
                        replacements = (
                            [ref]
                            if isinstance(ref, str) and ref in allowed
                            else aliases.get(ref, [ref]) if isinstance(ref, str) else [ref]
                        )
                        for replacement in replacements:
                            if replacement not in normalized:
                                normalized.append(replacement)
                    value[key] = normalized
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return payload


def _assert_payload_evidence_refs(
    payload: dict[str, Any], allowed_refs: list[str]
) -> None:
    """模型边界预检；未知证据不得被删除、替换或带入下游。"""
    allowed = set(allowed_refs)
    unknown: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_refs" and isinstance(child, list):
                    unknown.update(
                        ref for ref in child
                        if isinstance(ref, str) and ref not in allowed
                    )
                else:
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    if unknown:
        raise ValueError("模型输出包含证据白名单之外的引用")


class APICLINICRReasoner:
    """生产用 API 推理器：Step 1 候选生成与 Step 2 逐候选排除。"""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("L3_REASONER_MODEL") or get_model()

    def generate_candidates(self, profile: dict[str, Any]) -> CandidateGeneration:
        clinical_profile, allowed_refs = _prompt_profile(profile)
        evidence_catalog = _evidence_catalog(clinical_profile, allowed_refs)
        prompt = f"""你是心理评估系统 L3 的 CLINICR 鉴别诊断生成器。
只执行 Step 1，不下最终诊断，不选择疗法。
必须同时考虑以下10种障碍：{', '.join(SUPPORTED_DISORDERS)}。
根据患者画像生成3至5个不同候选，低置信度但临床上需要排除的候选也要保留。
只能使用输入画像中的事实；禁止补写患者未提供的症状。每条 evidence_refs 只能从“允许的 evidence_refs”列表中逐字复制；禁止自行拼接字段和值，禁止创造新引用。
confidence 是本步骤的主观初步概率，范围0至1。

只输出严格 JSON：
{{
  "considered_disorders": ["完整10种障碍代码"],
  "candidates": [
    {{
      "disorder": "MDD",
      "label": "重性抑郁障碍",
      "symptom_match": {{"matched": 5, "total": 9}},
      "core_evidence_summary": "支持候选的核心画像证据",
      "evidence_refs": ["PHQ9_item2=3"],
      "confidence": 0.75
    }}
  ]
}}

允许的 evidence_refs（只能复制 ref；target_path 和 patient_quote 仅帮助理解）：{json.dumps(evidence_catalog, ensure_ascii=False)}
患者画像：{json.dumps(clinical_profile, ensure_ascii=False)}
"""
        try:
            raw = ask_llm_stream(
                prompt,
                model=self.model,
                temperature=0.3,
                timeout=_l3_timeout_seconds(),
                extra_body=_l3_reasoning_extra(self.model),
            )
            payload = _canonicalize_payload_evidence_refs(
                _parse_json_object(raw), clinical_profile, allowed_refs
            )
            return CandidateGeneration.model_validate(payload)
        except APIError as exc:
            raise CLINICRAPIError(
                f"CLINICR Step 1 API错误：{type(exc).__name__}"
            ) from exc
        except Exception as exc:
            raise CLINICRModelUnavailable(
                f"CLINICR Step 1模型输出不合法：{type(exc).__name__}"
            ) from exc

    def evaluate_candidate(
        self,
        profile: dict[str, Any],
        candidate: DiagnosticCandidate,
    ) -> EliminativeAssessment:
        clinical_profile, allowed_refs = _prompt_profile(profile)
        evidence_catalog = _evidence_catalog(clinical_profile, allowed_refs)
        prompt = f"""你是心理评估系统 L3 的 ELIMINATIVE 逐项排除器。
只执行 Step 2：对指定候选逐条核对 DSM-5 / ICD-11 诊断标准。
只能使用患者画像中的事实，禁止把量表总分当成多个具体症状，禁止因“未提及”直接判定不符合。
status 规则：有明确支持证据为 met；有明确反证为 not_met；未提及或证据不足为 pending。
结论只能为：meets_criteria（满足完整诊断标准）、watch（部分满足/亚临床/仍待确认）、ruled_out（明确不满足核心标准）。
所有 met 或 not_met 条目必须携带证据；evidence_refs 只能从“允许的 evidence_refs”列表中逐字复制，禁止自行拼接或创造。pending 条目不得虚构证据，必须通过 missing_dimensions 给出可执行追问。

只输出严格 JSON：
{{
  "disorder": "{candidate.disorder}",
  "criteria_detail": [
    {{"code":"A1","name":"标准名称","status":"met","rationale":"判定理由","evidence_refs":["证据标识"]}}
  ],
  "conclusion": "meets_criteria|watch|ruled_out",
  "conclusion_reason": "结论依据",
  "missing_dimensions": [
    {{"field":"字段名","purpose":"为何需要","suggested_question":"建议L2追问"}}
  ]
}}

允许的 evidence_refs（只能复制 ref；target_path 和 patient_quote 仅帮助理解）：{json.dumps(evidence_catalog, ensure_ascii=False)}
患者画像：{json.dumps(clinical_profile, ensure_ascii=False)}
候选诊断：{candidate.model_dump_json()}
"""
        try:
            def request(step_prompt: str) -> str:
                return ask_llm_stream(
                    step_prompt,
                    model=self.model,
                    temperature=0,
                    timeout=_l3_timeout_seconds(),
                    extra_body=_l3_reasoning_extra(self.model),
                    response_format=JSON_OBJECT_RESPONSE_FORMAT,
                )

            def validate(raw: str) -> EliminativeAssessment:
                payload = _canonicalize_payload_evidence_refs(
                    _parse_json_object(raw), clinical_profile, allowed_refs
                )
                _assert_payload_evidence_refs(payload, allowed_refs)
                return EliminativeAssessment.model_validate(payload)

            repair_prompt = (
                prompt
                + "\n上一次回答不是符合上述结构的合法 JSON 对象，或包含证据白名单外的引用。"
                "请重新独立生成完整结果；只输出一个合法 JSON 对象，"
                "所有 evidence_refs 必须从允许列表逐字复制，"
                "不要解释、不要 Markdown、不要省略字段。"
            )
            return _request_structured_output_with_one_retry(
                request, validate, prompt, repair_prompt
            )
        except APIError as exc:
            raise CLINICRAPIError(
                f"CLINICR Step 2 API错误：{type(exc).__name__}"
            ) from exc
        except Exception as exc:
            raise CLINICRModelUnavailable(
                f"CLINICR Step 2模型输出不合法：{type(exc).__name__}"
            ) from exc


class APIRewardModelVerifier:
    """训练奖励模型上线前的 API 适配器；接口可直接替换为本地奖励模型。"""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("L3_VERIFIER_MODEL") or get_model()

    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        clinical_profile, allowed_refs = _prompt_profile(profile)
        evidence_catalog = _evidence_catalog(clinical_profile, allowed_refs)
        prompt = f"""你是 L3 的独立 Reward Model Verifier 临时API实现。
只评估排名第一诊断的推理过程质量，不重新生成诊断，也不知道候选生成器给出的置信度。
请评估：证据是否充分、逻辑是否连贯、是否考虑鉴别诊断、是否存在跳跃性推理。
score 范围0至1。四个 dimensions 必须各出现一次：evidence_sufficiency、logic_coherence、differential_coverage、no_reasoning_leaps。
若需要补充信息，missing_dimensions 必须给出具体字段、目的和供L2使用的追问；低分时 fail_reasons 必须解释原因。
所有 evidence_refs 只能从“允许的 evidence_refs”列表中逐字复制；没有对应证据时返回空列表，禁止创造引用。

只输出严格 JSON：
{{
  "score": 0.87,
  "dimensions": [
    {{"dimension":"evidence_sufficiency","score":0.9,"reason":"理由","evidence_refs":["证据标识"]}},
    {{"dimension":"logic_coherence","score":0.9,"reason":"理由","evidence_refs":[]}},
    {{"dimension":"differential_coverage","score":0.8,"reason":"理由","evidence_refs":[]}},
    {{"dimension":"no_reasoning_leaps","score":0.9,"reason":"理由","evidence_refs":[]}}
  ],
  "fail_reasons": [],
  "missing_dimensions": []
}}

允许的 evidence_refs（只能复制 ref；target_path 和 patient_quote 仅帮助理解）：{json.dumps(evidence_catalog, ensure_ascii=False)}
患者画像：{json.dumps(clinical_profile, ensure_ascii=False)}
待终验候选（不含候选置信度）：{json.dumps(candidate_without_confidence, ensure_ascii=False)}
逐项排除结果：{assessment.model_dump_json()}
其他候选的排除上下文：{json.dumps(differential_context, ensure_ascii=False)}
"""
        try:
            def request(step_prompt: str) -> str:
                return ask_llm_stream(
                    step_prompt,
                    model=self.model,
                    temperature=0,
                    timeout=_l3_timeout_seconds(),
                    extra_body=_l3_reasoning_extra(self.model),
                    response_format=JSON_OBJECT_RESPONSE_FORMAT,
                )

            def validate(raw: str) -> RewardModelAssessment:
                payload = _canonicalize_payload_evidence_refs(
                    _parse_json_object(raw), clinical_profile, allowed_refs
                )
                _assert_payload_evidence_refs(payload, allowed_refs)
                return RewardModelAssessment.model_validate(payload)

            repair_prompt = (
                prompt
                + "\n上一次回答不是符合上述结构的合法结果，或包含证据白名单外的引用。"
                "请重新独立生成完整 JSON；missing_dimensions 中必须使用字段名 "
                "field、purpose、suggested_question，禁止使用 suggested_question_for_L2。"
                "所有 evidence_refs 必须从允许列表逐字复制。"
                "只输出一个合法 JSON 对象，不要解释、不要 Markdown、不要省略字段。"
            )
            return _request_structured_output_with_one_retry(
                request, validate, prompt, repair_prompt
            )
        except APIError as exc:
            raise CLINICRAPIError(
                f"CLINICR Step 3 API错误：{type(exc).__name__}"
            ) from exc
        except Exception as exc:
            raise CLINICRModelUnavailable(
                f"CLINICR Step 3模型输出不合法：{type(exc).__name__}"
            ) from exc
