def crisis_response(risk_level, risk_types, sensitive_topics):
    result = {
        "crisis_flag": False,
        "flags": [],
        "instruction_to_next": ""
    }

    if risk_level >= 3:
        result["crisis_flag"] = True
        result["flags"].append("crisis_intervention")
        result["instruction_to_next"] += (
            "检测到高风险危机，停止普通心理流程，进入危机干预流程。"
        )

    elif "suicidal_ideation" in risk_types:
        result["flags"].append("monitor_suicidal_ideation")
        result["instruction_to_next"] += (
            "持续监测自杀相关想法，避免忽略风险变化。"
        )

    if "trauma" in sensitive_topics:
        result["flags"].append("avoid_trauma_probing")
        result["instruction_to_next"] += (
            "用户涉及创伤敏感话题，避免深度追问创伤细节，优先建立安全感。"
        )

    return result
