from copy import deepcopy
from typing import Any


CRISIS_RESPONSE_TEMPLATE = (
    "听起来你现在正承受非常强烈的痛苦。请先远离可能伤害自己的物品，"
    "尽快联系身边可信任的人陪伴你，并联系当地急救或危机干预服务。"
)

def execute_safety_policy(passport: dict[str, Any]) -> dict[str, Any]:
    """执行 L1 已作出的安全决定，不重新评估用户风险。"""
    decision = deepcopy(passport)
    flags = set(decision.get("flags") or [])
    labels = decision.get("structured_labels") or {}
    scale_skip_list = list(labels.get("scale_skip_list") or [])

    commands: list[str] = []
    if "avoid_trauma_probing" in flags:
        commands.append("avoid_trauma_probing")
    if "monitor_suicidal_ideation" in flags:
        commands.append("monitor_suicidal_ideation")
    for scale in scale_skip_list:
        commands.append(f"skip_scale:{scale}")

    crisis = decision.get("crisis_flag") is True
    if crisis:
        commands.extend(["stop_normal_pipeline", "do_not_call_l2", "show_crisis_response"])
        return {
            "route": "crisis",
            "terminal": True,
            "crisis_response_generated": True,
            "user_response": CRISIS_RESPONSE_TEMPLATE,
            "commands": list(dict.fromkeys(commands)),
            "applied_controls": list(dict.fromkeys(commands)),
        }

    return {
        "route": "l2",
        "terminal": False,
        "crisis_response_generated": False,
        "user_response": "",
        "commands": list(dict.fromkeys(commands)),
        "applied_controls": list(dict.fromkeys(commands)),
    }
