from __future__ import annotations

import re


def extract_explicit_scale_score(
    text: str, *, min_score: int, max_score: int
) -> int | None:
    """仅按患者明确给出的数字或固定选项解析量表答案。"""
    explicit = re.search(r"(?<!\d)([0-4])\s*分?(?!\d)", text)
    if explicit:
        score = int(explicit.group(1))
        return score if min_score <= score <= max_score else None

    normalized = text.strip()
    phrase_scores = (
        ("几乎每天", 3),
        ("一半以上", 2),
        ("有几天", 1),
        ("极其严重", 4),
        ("极重度", 4),
        ("相当严重", 3),
        ("重度", 3),
        ("中等", 2),
        ("中度", 2),
        ("轻微", 1),
        ("轻度", 1),
        ("完全没有", 0),
        ("从来没有", 0),
        ("严重", 3),
        ("没有", 0),
    )
    for phrase, score in phrase_scores:
        if phrase in normalized and min_score <= score <= max_score:
            return score

    exact = normalized.rstrip("。！？!? ")
    if exact in {"否", "不是", "没有", "没有过"} and min_score <= 0 <= max_score:
        return 0
    if exact in {"是", "是的", "有", "有过"} and min_score <= 1 <= max_score:
        return 1
    return None
