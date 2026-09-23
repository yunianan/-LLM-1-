from layers.layer2.schemas import L2SessionState
from shared.l2_l3_contract import missing_required_context_fields


def missing_dimensions(state: L2SessionState) -> list[str]:
    missing: list[str] = []
    if not state.focus_topics:
        missing.append("患者确认的当前重点")
    if len(state.symptoms) < 3:
        missing.append("至少3条有患者原话支持的关键症状")
    missing.extend(
        item.missing_label
        for item in missing_required_context_fields(state.diagnostic_context)
    )
    relevant = [
        item for item in state.scale_results.values()
        if item.status not in {"not_started", "skipped"}
    ]
    if relevant and not any(
        item.status in {"completed", "declined"} for item in relevant
    ):
        missing.append("相关量表完成或明确拒绝")
    if not state.patient_goals:
        missing.append("患者优先目标")
    if any(
        score is None for score in (
            state.motivation.importance,
            state.motivation.confidence,
            state.motivation.readiness,
        )
    ):
        missing.append("改变动机三项评分")
    return missing


def is_sufficient(state: L2SessionState) -> bool:
    return not missing_dimensions(state)
