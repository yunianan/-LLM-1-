from layers.layer4.config import Settings
from layers.layer4.service import L4Service, THERAPY_KEYWORDS


def _service() -> L4Service:
    return L4Service(Settings(
        _env_file=None,
        llm_provider="mock",
        review_mode="mock",
    ))


def test_validate_forbidden_words():
    service = _service()
    v = service._validate("你最好建议他辞职，我确诊了你的问题",
                          _turn(forbidden=["建议", "确诊"]), ["CBT"])
    assert any("建议" in x or "确诊" in x for x in v)


def test_validate_therapy_consistency_v43():
    service = _service()
    # MDD + CBT 方案下，回复提及 EMDR 属越界
    v = service._validate("我们可以尝试EMDR疗法", _turn(), ["CBT", "CBT-I"])
    assert any("EMDR" in x for x in v)
    # 允许的疗法不违规
    v2 = service._validate("我们试试CBT-I的睡眠限制", _turn(), ["CBT", "CBT-I"])
    assert v2 == []


def test_validate_allows_each_component_of_combined_base_therapy():
    service = _service()

    assert service._validate("我们使用SFBT聚焦当前问题", _turn(), ["CBT/SFBT"]) == []


def test_validate_absolute_bans():
    service = _service()
    v = service._validate("可以考虑开药治疗", _turn(), ["CBT"], ["禁止开药"])
    assert any("开药" in x for x in v)
    v2 = service._validate("我们先聊聊睡眠习惯", _turn(), ["CBT"], ["禁止开药"])
    assert v2 == []


def test_therapy_keywords_covered():
    assert "EMDR" in THERAPY_KEYWORDS
    assert "CBT-I" in THERAPY_KEYWORDS


def _turn(**kw):
    from layers.layer4.schemas.instructions import TurnInstruction
    d = dict(goal="g", technique="t")
    d.update(kw)
    return TurnInstruction(**d)
