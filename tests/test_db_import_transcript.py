"""``tools/db_import.py`` 的「会谈记录」读取路径。

这条路径全是启发式文本抽取（否认判定、话题归类、关键词切分），最容易悄悄判错，
所以单独锁住行为：判错的代价是生成一份看起来正常、实际上症状阴阳颠倒的病例。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from patient_agent.schemas import PatientScenario

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "db_import_under_test", PROJECT_ROOT / "tools" / "db_import.py"
)
assert _SPEC and _SPEC.loader
db_import: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(db_import)


def _transcript(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"conversation": [{"doctor": q, "patient": a} for q, a in pairs]}


# ------------------------------------------------------------------ 否认 / 肯定判定
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # 否定词落在后半句，只看句首会判反
        ("最近睡眠还可以，没有特别明显的入睡困难或者早醒问题。", "denial"),
        ("我一般胃口还不错，尤其是情绪稳定的时候。", "denial"),
        ("最近我倒是没有觉得胸闷或者呼吸不畅。", "denial"),
        ("头晕和头痛方面，其实我平时倒是没有特别明显的困扰。", "denial"),
        ("我平时的月经情况挺正常的，没有不规律或者特别疼痛的问题。", "denial"),
        ("没有。", "denial"),
        ("从来没有过。", "denial"),
        # 「从来没这么…过」是强调式，不是否认；前缀匹配「从来没」会把它判反。
        ("从来没这么想死过。", "affirmative"),
        ("我也不觉得绝望，就是有时候会感到无助。", "denial"),
        ("基本上没有出现过这种情况。", "denial"),
        # 有但轻仍算阳性：「偶尔」交给 severity 表达，不能吃掉症状
        ("偶尔可能因为加班或者没休息好会有点轻微的头晕，但并不是常态。", "affirmative"),
        ("心跳加快的情况倒是有时候会发生，特别是在我感到很紧张的时候。", "affirmative"),
        ("有时候我确实觉得思考变得有点慢，特别是在工作的时候容易走神。", "affirmative"),
        ("是的，这些强迫行为确实对我的生活有很大影响。", "affirmative"),
        ("有，我经常出现这种情况，几乎每天都要反复检查。", "affirmative"),
        ("是的，我父亲被诊断过精神分裂症，所以家族里确实有这方面的病史。", "affirmative"),
    ],
)
def test_classify_answer(answer: str, expected: str) -> None:
    assert db_import.classify_answer(answer) == expected


# ------------------------------------------------------------------ 来源识别
def test_only_conversation_shaped_records_are_tagged() -> None:
    records = db_import._tag_records(
        [
            {"record_id": "DB_1", "symptoms": {"depressed_mood": 2}},
            _transcript(("最近情绪怎么样？", "情绪还算稳定。")),
        ]
    )
    assert db_import._SOURCE_KIND_KEY not in records[0]
    assert records[1][db_import._SOURCE_KIND_KEY] == db_import.TRANSCRIPT_KIND
    assert records[1][db_import._SOURCE_INDEX_KEY] == 1


def test_load_source_tags_json_transcripts(tmp_path: Path) -> None:
    source = tmp_path / "transcripts.json"
    source.write_text(
        json.dumps([_transcript(("最近睡眠怎么样？", "睡眠还可以。"))], ensure_ascii=False),
        encoding="utf-8",
    )
    _, records = db_import.load_source(source, None)
    assert records[0][db_import._SOURCE_KIND_KEY] == db_import.TRANSCRIPT_KIND


def test_record_can_group_versions_of_one_patient() -> None:
    """同一名患者的多次会谈：靠记录里的 scenario_id / version 归成一个病例的多个版本。"""

    record = {
        **_transcript(("最近睡眠怎么样？", "睡眠还可以，没什么问题。")),
        "scenario_id": "S07_same_patient",
        "version": "2.0.0",
    }
    payload, _, _ = db_import.build_scenario_from_transcript(
        db_import._tag_records([record])[0], version=None, dataset_version=""
    )
    scenario = PatientScenario.model_validate(payload)
    assert scenario.scenario_id == "S07_same_patient"
    assert scenario.version == "2.0.0"


# ------------------------------------------------------------------ 整条抽取
@pytest.fixture
def extracted() -> tuple[dict[str, Any], list[str], list[str]]:
    return db_import.build_scenario_from_transcript(
        db_import._tag_records(
            [
                _transcript(
                    (
                        "你好，最近有没有感觉特别焦虑或者情绪低落？",
                        "最近情绪还算稳定，但有时候还是会感到焦虑和紧张。"
                        "我做事情总是反复检查，生怕出错，这让我很疲惫。",
                    ),
                    ("这些反复检查有没有影响到你的工作或者生活？", "是的，确实影响很大。"),
                    ("最近睡眠怎么样？有没有入睡困难或者早醒？", "睡眠还可以，没有特别明显的入睡困难。"),
                    ("有没有过想伤害自己或者自杀的念头？", "我没有过想伤害自己或者自杀的念头。"),
                    (
                        "有没有过持续的兴奋或者特别亢奋的时期？",
                        "没有持续的兴奋或者特别亢奋的时期。",
                    ),
                    (
                        "你们家族里有没有人有过类似的精神健康问题？",
                        "是的，我父亲被诊断过精神分裂症，家族里确实有这方面的病史。",
                    ),
                    ("平时有什么让你感到压力的事情吗？", "说不清，就是一种说不上来的感觉，心里总堵着一块。"),
                )
            ]
        )[0],
        version=None,
        dataset_version="",
    )


def test_extracted_scenario_is_schema_valid(extracted) -> None:
    payload, _, _ = extracted
    scenario = PatientScenario.model_validate(payload)
    assert scenario.scenario_id == "TRANSCRIPT_00"
    assert scenario.version == "0.1.0-draft"
    assert "needs_review" in scenario.tags


def test_opening_is_the_real_first_patient_turn(extracted) -> None:
    payload, _, _ = extracted
    opening = payload["disclosure"]["opening_templates"][0]
    assert opening == payload["clinical"]["chief_complaint"]
    assert opening.startswith("最近情绪还算稳定")


def test_symptoms_split_into_present_and_absent(extracted) -> None:
    payload, _, _ = extracted
    by_id = {item["symptom_id"]: item for item in payload["clinical"]["symptoms"]}
    # 焦虑虽在否定分句之后出现，也必须是阳性
    assert by_id["anxiety"]["present"] is True
    assert by_id["obsession_compulsion"]["present"] is True
    # 睡眠那句是否认，不能变成阳性症状
    assert by_id["insomnia"]["present"] is False
    assert by_id["insomnia"]["denial_utterance"].startswith("睡眠还可以")
    assert sum(1 for item in by_id.values() if item["present"]) == 2


def test_chief_complaint_symptom_is_proactive(extracted) -> None:
    payload, _, _ = extracted
    proactive = [
        item["symptom_id"]
        for item in payload["clinical"]["symptoms"]
        if item.get("disclosure") == "proactive"
    ]
    # 开场陈述里提到的症状必须有一条主动说出，否则前几轮系统识别不到
    assert len(proactive) == 1
    assert any(
        trigger in payload["clinical"]["chief_complaint"]
        for trigger in db_import._TOPIC_TRIGGERS_BY_CODE[proactive[0]]
    )


def test_denials_become_absent_exclusions(extracted) -> None:
    payload, _, _ = extracted
    by_id = {item["condition_id"]: item for item in payload["clinical"]["exclusions"]}
    assert by_id["manic_history"]["present"] is False
    assert by_id["manic_history"]["denial_utterance"]
    # risk 的否认句同样要落地，否则 Schema 直接拒绝 level=none 的病例
    assert payload["risk"]["level"] == "none"
    assert "自杀" in payload["risk"]["denial_utterance"]


def test_family_history_label_is_whitelisted(extracted) -> None:
    payload, warnings, _ = extracted
    # 守卫只认「医生说 / 医院说」这类转述前缀，认不出「我父亲被诊断为」，
    # 所以必须把家族史里的诊断名加进白名单，否则患者一开口就被判越界。
    assert payload["clinical"]["patient_reported_labels"] == ["精神分裂症"]
    assert any("精神分裂症" in item for item in warnings)


def test_unclassifiable_answer_is_parked_and_flagged(extracted) -> None:
    payload, _, todos = extracted
    parked = [fact for fact in payload["background"]["facts"] if fact["field"] == db_import.RAW_ANSWER_FIELD]
    assert parked and parked[0]["content"].startswith("说不清")
    assert any("没能归类" in item for item in todos)


def test_expected_is_expanded_and_draft_todos_are_emitted(extracted) -> None:
    payload, _, todos = extracted
    # 必须整块展开：--baseline 会直接索引 expected["events"]
    assert set(payload["expected"]) == {
        "termination_reason",
        "events",
        "forbidden_events",
        "expected_excluded_conditions",
        "expected_layers",
        "min_turns",
    }
    assert any("scale_plans 为空" in item for item in todos)
    assert any("expected / stop 全部留空" in item for item in todos)


def test_scenario_without_positive_symptom_still_validates() -> None:
    payload, _, todos = db_import.build_scenario_from_transcript(
        db_import._tag_records(
            [_transcript(("最近睡眠怎么样？", "睡眠还可以，没什么问题。"))]
        )[0],
        version="1.0.0",
        dataset_version="",
    )
    scenario = PatientScenario.model_validate(payload)
    present = [item for item in scenario.clinical.symptoms if item.present]
    assert [item.symptom_id for item in present] == ["depressed_mood"]
    assert present[0].utterance == payload["clinical"]["chief_complaint"]
    assert any("没有可判定的阳性症状" in item for item in todos)


# ------------------------------------------------------------------ 关键词切分
def test_probe_keywords_strip_question_fillers() -> None:
    keywords = db_import.extract_probe_keywords(
        "最近有没有觉得情绪低落，或者对以前喜欢做的事情提不起兴趣？"
    )
    assert "情绪低落" in keywords
    assert all("有没有" not in item for item in keywords)
