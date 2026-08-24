CRISIS_WORDS = {
    "自杀": 3,
    "结束生命": 3,
    "不想活了": 3,
    "活着没意义": 2,
    "割腕": 3,
    "伤害自己": 2,
    "杀了他": 3,
    "报复": 2,
    "被家暴": 2,
    "被虐待": 2,
    "被性侵": 2,
}

RISK_TYPE_BY_WORD = {
    "自杀": "suicidal_ideation",
    "结束生命": "suicidal_ideation",
    "不想活了": "suicidal_ideation",
    "活着没意义": "suicidal_ideation",
    "割腕": "self_harm",
    "伤害自己": "self_harm",
    "杀了他": "violence",
    "报复": "violence",
    "被家暴": "abuse",
    "被虐待": "abuse",
    "被性侵": "abuse",
}

SENSITIVE_TOPIC_WORDS = {
    "trauma": ["创伤", "童年", "小时候", "家暴", "虐待", "性侵", "事故"],
    "bereavement": ["去世", "丧亲", "离开人世"],
    "despair": ["绝望"],
}


def keyword_detect(text):
    max_score = 0
    matched_words = []
    risk_types = []
    sensitive_topics = []
    evidence_refs = []

    for word, score in CRISIS_WORDS.items():
        if word in text:
            matched_words.append(word)
            if score > max_score:
                max_score = score
            rt = RISK_TYPE_BY_WORD.get(word)
            if rt and rt not in risk_types:
                risk_types.append(rt)
            evidence_refs.append({
                "source": "L1_rule_engine",
                "finding": f"matched_keyword={word}",
                "basis": f"关键词匹配风险模式:{word}"
            })

    for topic, words in SENSITIVE_TOPIC_WORDS.items():
        if any(w in text for w in words):
            sensitive_topics.append(topic)
            evidence_refs.append({
                "source": "L1_rule_engine",
                "finding": f"sensitive_topic={topic}",
                "basis": f"关键词匹配敏感主题:{topic}"
            })

    return {
        "rule_score": max_score,
        "risk_keywords": matched_words,
        "risk_types": risk_types,
        "sensitive_topics": sensitive_topics,
        "evidence_refs": evidence_refs
    }
