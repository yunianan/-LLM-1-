import json
from datetime import datetime, timezone

from shared.llm_client import ask_llm
from .harm_checker import generate_harm_rules
from .boundary_checker import generate_boundary_rules
from .quality_checker import generate_quality_rules

REVIEW_PROMPT_TEMPLATE = """你是输出安全审核系统。对以下治疗回复同时完成三项检查：

1. 危害性(harm)：是否存在危险建议、自伤鼓励、不当用药、高危回应失当？
2. 越界(boundary)：是否诊断新疾病、替代医生、预测未来、超出能力？
3. 质量(quality)：是否疗法一致、逻辑连贯、无极端词汇？

上下文条件与动态审核规则：
{context_rules}

待审核AI回复：
{l4_output}

请严格输出JSON，不要输出其他内容：
{{
"harm": "pass|fail",
"boundary": "pass|fail",
"quality": "pass|fail",
"fail_reason": "",
"fix_instruction": ""
}}
"""


def _dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def build_review_prompt(l4_output, harm_rules, boundary_rules, quality_rules):
    context_rules = _dump({
        "harm_rules": harm_rules,
        "boundary_rules": boundary_rules,
        "quality_rules": quality_rules,
    })
    return REVIEW_PROMPT_TEMPLATE.format(
        l4_output=l4_output,
        context_rules=context_rules,
    )


def review_response(ctx):
    l4_output = ctx["l4_raw_output"]
    l4_substage = ctx.get("l4_substage", "L4-L1")
    therapy_type = ctx.get("therapy_type", "")
    diagnosis_context = ctx.get("diagnosis_context") or {}
    risk_context = ctx.get("risk_context") or {}

    harm_rules = generate_harm_rules(diagnosis_context, risk_context)
    boundary_rules = generate_boundary_rules(diagnosis_context, therapy_type, l4_substage)
    quality_rules = generate_quality_rules(therapy_type, l4_substage, diagnosis_context)

    prompt = build_review_prompt(l4_output, harm_rules, boundary_rules, quality_rules)
    response = ask_llm(prompt)

    try:
        data = json.loads(response)
    except Exception:
        data = {
            "harm": "fail",
            "boundary": "fail",
            "quality": "fail",
            "fail_reason": "审核模型输出解析失败，为安全起见按不通过处理",
            "fix_instruction": "请重写回复：保持共情，使用非确定性措辞，避免任何诊断或医疗建议。",
        }

    harm = data.get("harm", "fail")
    boundary = data.get("boundary", "fail")
    quality = data.get("quality", "fail")
    fail_reason = data.get("fail_reason", "")
    fix_instruction = data.get("fix_instruction", "")

    passed = harm == "pass" and boundary == "pass" and quality == "pass"

    evidence_refs = []
    for rule in harm_rules:
        evidence_refs.append({
            "rule_id": rule["rule_id"],
            "dimension": "harm",
            "status": "pass" if harm == "pass" else "checked",
            "rule": rule["check"],
        })
    for rule in boundary_rules:
        evidence_refs.append({
            "rule_id": rule["rule_id"],
            "dimension": "boundary",
            "status": "pass" if boundary == "pass" else "checked",
            "rule": rule["check"],
        })
    for rule in quality_rules:
        evidence_refs.append({
            "rule_id": rule["rule_id"],
            "dimension": "quality",
            "status": "pass" if quality == "pass" else "checked",
            "rule": rule["check"],
        })

    return {
        "layer_id": "L5",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "harm": harm,
        "boundary": boundary,
        "quality": quality,
        "fail_reason": fail_reason,
        "fix_instruction": fix_instruction,
        "evidence_refs": evidence_refs,
        "confidence": 0.95 if passed else 0.9,
        "flags": [] if passed else ["manual_review_required"],
        "instruction_to_next": None,
    }
