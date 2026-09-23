import json
import re

from pydantic import BaseModel, ConfigDict, Field

from shared.llm_client import ask_llm
from shared.prompts.risk_prompt import build_prompt


MAX_ATTEMPTS = 2


class RiskClassification(BaseModel):
    """L1 风险模型的严格输出契约；拒绝类型转换和未知字段。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    risk_level: int = Field(ge=0, le=3)
    score: int = Field(ge=0, le=3)
    risk_types: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def _extract_json(response: str) -> str:
    if not isinstance(response, str):
        raise TypeError("LLM响应必须是字符串")
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return text


def _fallback(error: Exception, attempts: int) -> dict:
    return {
        "llm_score": 2,
        "risk_types": ["unclassified_risk"],
        "confidence": 0.0,
        "degraded": True,
        "failure_reason": f"{type(error).__name__}: {error}",
        "attempts": attempts,
        "evidence_refs": [{
            "source": "L1_LLM_fallback",
            "finding": "llm_classification_unavailable",
            "basis": "LLM风险分类失败，采用保守阻断并要求人工复核",
        }],
    }


def llm_classify(text):
    prompt = build_prompt(text)
    last_error: Exception = RuntimeError("LLM风险分类未执行")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = ask_llm(prompt)
            data = RiskClassification.model_validate(
                json.loads(_extract_json(response))
            )
            return {
                # risk_level 与 score 均由提示词要求；不一致时保守取较高值。
                "llm_score": max(data.score, data.risk_level),
                "risk_types": data.risk_types,
                "confidence": data.confidence,
                "degraded": False,
                "failure_reason": "",
                "attempts": attempt,
                "evidence_refs": [{
                    "source": "DeepSeek_API",
                    "finding": f"risk_level={data.risk_level}",
                    "basis": data.reason,
                }],
            }
        except Exception as exc:
            # 网络 SDK 会抛出自身的连接/超时异常类型；统一降级，绝不让安全入口崩溃。
            last_error = exc

    return _fallback(last_error, MAX_ATTEMPTS)
