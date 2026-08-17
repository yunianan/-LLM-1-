from l4.service import L4Service, THERAPY_KEYWORDS


def test_validate_forbidden_words():
    service = L4Service()
    v = service._validate("你最好建议他辞职，我确诊了你的问题",
                          _turn(forbidden=["建议", "确诊"]), ["CBT"])
    assert any("建议" in x or "确诊" in x for x in v)


def test_validate_therapy_consistency_v43():
    service = L4Service()
    # MDD + CBT 方案下，回复提及 EMDR 属越界
    v = service._validate("我们可以尝试EMDR疗法", _turn(), ["CBT", "CBT-I"])
    assert any("EMDR" in x for x in v)
    # 允许的疗法不违规
    v2 = service._validate("我们试试CBT-I的睡眠限制", _turn(), ["CBT", "CBT-I"])
    assert v2 == []


def test_therapy_keywords_covered():
    assert "EMDR" in THERAPY_KEYWORDS
    assert "CBT-I" in THERAPY_KEYWORDS


def _turn(**kw):
    from l4.schemas.instructions import TurnInstruction
    d = dict(goal="g", technique="t")
    d.update(kw)
    return TurnInstruction(**d)
