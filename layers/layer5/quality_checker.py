def generate_quality_rules(therapy_type, l4_substage, diagnosis_context=None):
    rules = []

    if therapy_type == "MI":
        rules.append({
            "rule_id": "R-QUALITY-MI-001",
            "dimension": "empathy",
            "condition": "MI阶段需要体现共情和理解",
            "check": "检查回复是否承认用户感受，而不是简单评价或指导"
        })
        rules.append({
            "rule_id": "R-QUALITY-MI-002",
            "dimension": "reflection",
            "condition": "MI需要反映用户表达",
            "check": "检查回复是否包含对用户观点或情绪的反映"
        })
        rules.append({
            "rule_id": "R-QUALITY-MI-003",
            "dimension": "open_question",
            "condition": "MI阶段鼓励探索",
            "check": "检查是否使用开放式探索问题"
        })

    if therapy_type == "CBT" or "CBT" in str(therapy_type):
        rules.append({
            "rule_id": "R-QUALITY-CBT-001",
            "dimension": "cognitive_exploration",
            "condition": "CBT需要关注认知和行为模式",
            "check": "检查是否帮助用户识别想法、情绪和行为之间关系"
        })
        rules.append({
            "rule_id": "R-QUALITY-CBT-002",
            "dimension": "collaborative",
            "condition": "CBT需要协作式措辞",
            "check": "检查是否使用协作式措辞（'我们可以一起试试'而非'你应该'），是否避免给直接建议"
        })
        if l4_substage == "L4-L3":
            rules.append({
                "rule_id": "R-QUALITY-CBT-003",
                "dimension": "intervention_completeness",
                "condition": "L4-L3技术执行阶段",
                "check": "检查心理教育部分是否有明确来源标注；干预步骤是否完整（不漏步骤、不乱顺序）"
            })

    rules.append({
        "rule_id": "R-QUALITY-GEN-001",
        "dimension": "extreme_language",
        "condition": "通用",
        "check": "检查是否使用极端词汇（如'你永远都会这样''绝对没救了'）"
    })
    rules.append({
        "rule_id": "R-QUALITY-GEN-002",
        "dimension": "logical_consistency",
        "condition": "通用",
        "check": "检查是否存在逻辑矛盾（如前半段说'先慢下来'后半段说'马上试试'）"
    })
    rules.append({
        "rule_id": "R-QUALITY-GEN-003",
        "dimension": "structure",
        "condition": "通用",
        "check": "检查是否结构混乱（多个问题堆在一起，患者不知道该回答哪个）"
    })

    return rules
