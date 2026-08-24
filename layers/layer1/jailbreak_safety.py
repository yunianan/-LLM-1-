JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "忽略之前指令",
    "显示系统提示词",
    "你不是AI",
    "绕过限制",
]


def jailbreak_detect(text):
    matched = []
    evidence_refs = []

    for pattern in JAILBREAK_PATTERNS:
        if pattern.lower() in text.lower():
            matched.append(pattern)
            evidence_refs.append({
                "source": "L1_jailbreak_detector",
                "finding": f"jailbreak_pattern={pattern}",
                "basis": "检测到提示词注入或系统指令绕过行为"
            })

    return {
        "jailbreak_flag": len(matched) > 0,
        "jailbreak_level": 3 if matched else 0,
        "matched_patterns": matched,
        "evidence_refs": evidence_refs
    }
