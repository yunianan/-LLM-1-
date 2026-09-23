"""指令执行守卫：核验上游授权、轮次和证据，不进行临床推理。"""

import re

from shared.schemas.treatment_instructions import TreatmentInstructions, TreatmentTurnRecord

STAGES = ("L4-L1", "L4-L2", "L4-L3", "L4-L4")


def turn_sources(records: list[TreatmentTurnRecord], message: str) -> dict[str, str]:
    sources = {f"turn:{len(records) + 1}:user": message}
    for i, record in enumerate(records, 1):
        sources[f"turn:{i}:user"] = record.user_message
        sources[f"turn:{i}:assistant"] = record.reply
    return sources


def validate_turn_instructions(
    bundle: TreatmentInstructions,
    *,
    session_id: str,
    user_message: str,
    records: list[TreatmentTurnRecord],
    safety: dict | None = None,
) -> None:
    safety = safety or {}
    if safety.get("risk_level", 0) >= 3 or safety.get("crisis_flag") or safety.get("safety_flag") == "block":
        raise ValueError("L1 安全控制不允许普通治疗")
    turn = bundle.L3
    if turn is None:
        raise ValueError("缺少本轮 L3 动态指令")
    if (turn.session_id != session_id or turn.turn_number != len(records) + 1
            or turn.source_message != user_message):
        raise ValueError("动态指令与当前会话、轮次或患者消息不匹配")
    if not turn.goal.strip() or not turn.rationale.strip():
        raise ValueError("动态指令缺少目标或依据")
    anchor = bundle.L2
    if (turn.technique_code not in anchor.allowed_techniques
            or turn.technique != anchor.allowed_techniques[turn.technique_code]):
        raise ValueError("本轮技术不在 L3 授权白名单内")
    if not set(bundle.L4.absolute_bans) <= set(turn.forbidden):
        raise ValueError("本轮指令削弱了防越界要求")
    if not set(turn.plugin_steps) <= set(anchor.plugins):
        raise ValueError("本轮使用了未激活的插件")
    expected_guidance = "；".join(f"{name}：{step}" for name, step in turn.plugin_steps.items()) or None
    if turn.plugin_guidance != expected_guidance:
        raise ValueError("插件文本与结构化插件指令不一致")
    if any(not step.strip() for step in turn.plugin_steps.values()):
        raise ValueError("插件嵌入指导为空")
    action_text = " ".join([turn.goal, *turn.plugin_steps.values()])
    if any(ban in action_text for ban in bundle.L4.absolute_bans if ban):
        raise ValueError("本轮执行文本包含绝对禁止行为；限制应写在 forbidden，不应写成治疗目标")
    # 文本不能借用自由字段启用未授权插件；语义质量仍由模型和 L5 复核。
    text = " ".join([turn.goal, turn.link_previous, turn.plugin_guidance or ""])
    for plugin in ("EMDR", "ERP", "MBCT", "ACT", "IPT", "IPSRT", "SFBT", "CBT-I"):
        if (re.search(rf"(?<![A-Za-z]){re.escape(plugin)}(?![A-Za-z])", text, re.I)
                and plugin not in anchor.plugins
                and plugin not in anchor.therapy_type.replace("/", "+").split("+")):
            raise ValueError(f"指令文本包含未授权疗法：{plugin}")
    adjustment = anchor.strategy_adjustments
    if adjustment.get("challenge_interventions") == "deferred" and any(
        term in action_text for term in ("暴露", "行为实验", "创伤加工", "EMDR", "ERP")
    ):
        raise ValueError("本轮执行文本违反暂缓挑战性干预要求")
    if (adjustment.get("motivation_band") == "low"
            or adjustment.get("care_priority", "psychotherapy") != "psychotherapy") and turn.plugin_steps:
        raise ValueError("当前需动机唤出、稳定化或转介，不能插入行动导向插件")
    if adjustment.get("challenge_interventions") == "deferred" and set(turn.plugin_steps) & {"ERP", "EMDR"}:
        raise ValueError("当前暂缓挑战性干预")
    if any("暴露" in item for item in anchor.contraindications):
        if set(turn.plugin_steps) & {"ERP", "EMDR"} or any(word in text for word in ("暴露", "创伤加工")):
            raise ValueError("本轮指令违反暴露禁忌")
    if "trauma" in safety.get("sensitive_topics", []):
        if (turn.technique_code in {"trauma_processing", "emdr", "exposure_response_prevention"}
                or set(turn.plugin_steps) & {"EMDR", "ERP"}
                or any(word in text for word in ("暴露", "创伤加工"))):
            raise ValueError("本轮指令违反最新 L1 创伤限制")
    if adjustment.get("technical_density") in {"reduced", "light"} and len(turn.plugin_steps) > 1:
        raise ValueError("低技术密度时不得在本轮叠加多个插件")
    sources = turn_sources(records, user_message)
    if not turn.evidence:
        raise ValueError("本轮指令没有对话证据")
    for ref in turn.evidence + [item.evidence for item in turn.observations]:
        if not ref.quote.strip() or ref.source_id not in sources or ref.quote not in sources[ref.source_id]:
            raise ValueError("本轮指令引用了不存在的原话或对话轮次")
    current_source = f"turn:{len(records) + 1}:user"
    if not any(ref.source_id == current_source for ref in turn.evidence):
        raise ValueError("本轮指令未引用当前患者消息")
    if records and not turn.link_previous.strip():
        raise ValueError("多轮指令缺少上下文衔接")
    previous = next((record.instruction.force_substage for record in reversed(records) if not record.fallback_used), "L4-L1")
    if turn.force_substage not in STAGES:
        raise ValueError("缺少 L3 指定的执行阶段")
    before, after = STAGES.index(previous), STAGES.index(turn.force_substage)
    if not records and after != 0:
        raise ValueError("首轮必须从建立治疗联盟开始")
    if after < 2 and turn.technique_code not in {"reflection", "feedback", "mi_evoking", "stabilization", "review"}:
        raise ValueError("共情与反馈阶段不能提前执行治疗技术")
    if after > before:
        if (after != before + 1 or not turn.previous_goal_met or not turn.patient_ready
                or not records or records[-1].fallback_used):
            raise ValueError("阶段推进缺少前一目标完成及患者准备证据，或发生跳级")
