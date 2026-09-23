from __future__ import annotations

import json
import os

from shared.llm_client import ask_llm_stream

from .models import SafetyDisposition, SupportResponse
from .risk_classifier import _json_object


FALLBACKS: dict[SafetyDisposition, str] = {
    "emergency_evaluation": (
        "谢谢您愿意告诉我这些。现在最重要的是确保您不必独自面对危险。"
        "普通评估已经暂停，请立即联系能够陪在您身边的人，并按照下方安全步骤寻求紧急帮助。"
    ),
    "further_evaluation": (
        "谢谢您坦诚说明这些经历。暂停普通评估不是拒绝帮助，而是为了先更可靠地确认您的安全。"
        "请按照下方步骤联系专业人员，并尽量让可信任的人陪伴您。"
    ),
    "non_urgent_follow_up": (
        "谢谢您认真完成这些安全问题。当前更适合先落实安全计划和后续支持，"
        "您不需要独自承担这些感受；完成下面的安排后，我们再决定如何继续。"
    ),
    "no_further_safety_intervention": "谢谢您完成安全评估。我们已经记录相关信息，可以继续了解您当前最需要的帮助。",
    "not_assessed": "谢谢您愿意说明现在的情况。",
}


PROHIBITED = ("想开点", "你不会真的", "保证没事", "都是为了家人", "必须坚强")


def generate_support(
    *,
    patient_quote: str,
    disposition: SafetyDisposition,
    required_action: str,
) -> tuple[str, str]:
    prompt = f"""
你负责生成安全过渡阶段的个性化中文支持回应。根据患者原话承认其感受、感谢披露，
并自然衔接已经确定的行动。不得改变处置、诊断、说教、施加内疚、承诺绝对安全、
淡化风险、描述自伤方法或延迟求助。输出60至180字。
只输出符合JSON Schema的JSON对象。

患者原话：{patient_quote}
处置：{disposition}
必须衔接的行动：{required_action}
JSON Schema：{json.dumps(SupportResponse.model_json_schema(), ensure_ascii=False)}
""".strip()
    try:
        timeout = float(os.getenv("L1_SUPPORT_TIMEOUT_SECONDS", "12"))
        raw = ask_llm_stream(
            prompt,
            timeout=timeout,
            response_format={"type": "json_object"},
        )
        message = SupportResponse.model_validate(_json_object(raw)).supportive_message.strip()
        if any(item in message for item in PROHIBITED):
            raise ValueError("支持话术包含禁止表达")
        if required_action and not any(
            token in message for token in ("安全", "帮助", "专业", "联系", "支持", "暂停")
        ):
            raise ValueError("支持话术没有衔接安全行动")
        return message, "llm"
    except Exception:
        return FALLBACKS[disposition], "fallback"

