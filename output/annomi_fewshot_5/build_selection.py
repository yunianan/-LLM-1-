"""Export five complete AnnoMI records, with separately authored teaching notes."""
import csv
import io
import json
from datetime import date
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/uccollab/AnnoMI/main/"
SELECTION = [
    (23, "初次访谈：焦虑及生活影响", "参与、聚焦", [
        ([2, 16], "了解来访背景与就诊感受", "开放提问、情绪反映", "先承接求助体验，再了解近期变化。"),
        ([17, 32], "探索功能影响与具体情境", "聚焦、开放提问", "从笼统焦虑转到工作和社交中的实例。"),
        ([33, 78], "了解情境、想法、身体反应及应对行为", "具体化追问、总结", "把来访者描述整理为可核对的信息，不提前输出诊断。"),
    ], "视频标题明确为 CBT 示范；用于访谈探索，不当成纯 MI 或诊断标准。记录止于一次情境回顾，不是完成评估的范本。不要照搬视频中的观察性描述，例如声称看见患者焦虑。"),
    (43, "考试焦虑：探索改变信心", "唤出、计划", [
        ([0, 21], "理解想改变与回避之间的矛盾", "反映、信心尺", "承认焦虑和改变意愿可以同时存在。"),
        ([22, 38], "发现既往成功经验", "开放提问、肯定", "让来访者提供自己的能力依据。"),
        ([39, 49], "协商可行下一步", "唤出、总结", "由来访者提出增加信心的行动。"),
    ], "包含较早的 1—10 分信心尺，它不是 PHQ-9/ISI 等症状量表，不得用它学习症状量表触发时机。原文存在转录不流畅之处，未擅自纠正。"),
    (47, "暂不愿开口：尊重自主权", "参与", [
        ([0, 7], "识别暂不愿交流", "澄清", "对拒绝表达保持敏感，不逼问。"),
        ([8, 11], "表达倾听意愿并尊重选择", "支持、自主权确认", "允许来访者暂时不说。"),
        ([12, 14], "结束当前接触", "协商后续联系", "仅借鉴允许暂停的做法，不承诺系统无法执行的联系。"),
    ], "这是短篇示范，不足以代表完整评估。第 6 条关于对方参与争吵的猜测不宜模仿；第 12 条上门和电话随访承诺不适用于当前系统。完整保留供核对，正向 few-shot 优先使用第 8—11 条，整段不可无条件作为正例。"),
    (119, "求职与育儿：厘清现实障碍", "聚焦、唤出、计划", [
        ([0, 19], "了解近况与失望", "情绪反映、重要性及信心探索", "先了解阻碍，不把现实困难解释为缺乏动机。"),
        ([20, 69], "明确目标和可行条件", "开放提问、澄清", "区分托育、技能和自信问题。"),
        ([70, 85], "总结并让来访者确定优先事项", "总结、自主选择", "由来访者决定先解决哪个实际障碍。"),
    ], "属于复诊语境，不能照搬成首次会谈开场；包含重要性/信心尺。原始记录结尾仍是未完成的话语，已全部保留，不补造结尾。"),
    (22, "价值目标：家庭与自我照顾", "聚焦、唤出", [
        ([0, 71], "了解来访者重视的价值及变化", "澄清、反映、肯定", "遵循来访者自己的价值，不植入咨询师价值判断。"),
        ([72, 111], "探索价值与日常投入的差距", "聚焦、反映", "了解照顾家庭和照顾自己的张力。"),
        ([112, 132], "承接压力并总结", "倾听、总结", "听到睡眠困难后仍结合上下文理解，没有立即插入 ISI。"),
    ], "开场已有价值澄清表及既往咨询背景，不适合直接作为首次开场；宗教内容只适用于来访者主动提出的价值语境。咨询师部分封闭或引导性问题不应机械复制。"),
]


def fetch(name):
    local = ROOT / name
    if local.exists():
        return list(csv.DictReader(io.StringIO(local.read_text(encoding="utf-8-sig"))))
    with urlopen(BASE + name, timeout=60) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))


def main():
    simple, full = fetch("AnnoMI-simple.csv"), fetch("AnnoMI-full.csv")
    result = []
    for ident, title, stages, notes, caution in SELECTION:
        rows = sorted((r for r in simple if int(r["transcript_id"]) == ident), key=lambda r: int(r["utterance_id"]))
        assert rows and all(r["mi_quality"] == "high" for r in rows)
        ids = [int(r["utterance_id"]) for r in rows]
        assert len(ids) == len(set(ids)) and ids == list(range(len(ids)))
        messages = []
        for r in rows:
            annotations = [a for a in full if a["transcript_id"] == str(ident) and a["utterance_id"] == r["utterance_id"]]
            assert annotations and all(a["utterance_text"] == r["utterance_text"] for a in annotations)
            messages.append({"utterance_id": int(r["utterance_id"]), "role": "assistant" if r["interlocutor"] == "therapist" else "user", "interlocutor": r["interlocutor"], "content": r["utterance_text"], "timestamp": r["timestamp"], "main_therapist_behaviour": r["main_therapist_behaviour"], "client_talk_type": r["client_talk_type"], "expert_annotations": [{k: v for k, v in a.items() if k in ("annotator_id", "therapist_input_exists", "therapist_input_subtype", "reflection_exists", "reflection_subtype", "question_exists", "question_subtype")} for a in annotations]})
        result.append({"transcript_id": ident, "title_zh": title, "mi_quality": "high", "topic": rows[0]["topic"], "video_title": rows[0]["video_title"], "video_url": rows[0]["video_url"], "utterance_count": len(rows), "completeness": "all utterances of this transcript in AnnoMI; not necessarily a complete clinical session or course", "suggested_stages_zh": stages, "teaching_notes_origin": "assistant-authored observable technique summaries; not dataset stage labels or original therapist reasoning", "teaching_notes": [{"utterance_range": span, "goal_zh": goal, "technique_zh": technique, "rationale_summary_zh": summary} for span, goal, technique, summary in notes], "usage_caution_zh": caution, "messages": messages})
    payload = {"source": "https://github.com/uccollab/AnnoMI", "retrieved_on": date.today().isoformat(), "language": "en", "annotation_language": "zh", "license_note": "Public availability is not proof of unrestricted reuse; confirm source permissions before redistribution or deployment.", "dialogues": result}
    (ROOT / "selected_dialogues.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# AnnoMI：五段完整对话及 few-shot 使用说明", "", "原始英文逐句保留；中文为另外编写的教学标注。共五段、363 条发言。完整指收录原数据中该 transcript 的全部发言，不意味着完整疗程或四阶段全部完成。", "", "## 使用方式", "", "- JSON 的 messages 保留全部发言及原始专家标注，按 utterance_id 排序；相同发言的多个标注者放在 expert_annotations，避免重复发言。", "- 中文阶段、目标、技巧和简短依据是整理者标注，不是数据集标签，也不是咨询师内心思维链。可作为独立的教学说明，不作为患者可见回复。", "- 五段作为完整候选资料库；每次按场景选择相关样本，不建议每轮同时注入全部内容。不能将对话事实写入当前患者画像。", "- 第 47 段包含不宜模仿的句子，保留完整原文供审阅；未经审核不能整段用作正例。", "- 不根据示范中的打分或表单话术修改现有量表许可规则。数据集没有提供量表触发的临床标准。", "- 原始视频为示范；高质量是整段 MI 标签，不等于每句适用于本系统。公开可读取不等于无限制再发布，正式使用前核实授权。", "", "来源：https://github.com/uccollab/AnnoMI", "", "## 数据文件", "", "- selected_dialogues.json：五段完整原文、来源、专家标注及中文教学说明。", "- 本文：适合阅读的完整原文和使用边界。", ""]
    for d in result:
        lines.extend([f"## ID {d['transcript_id']} · {d['title_zh']}", "", f"发言数：{d['utterance_count']}；建议参考阶段：{d['suggested_stages_zh']}。", "", f"原视频：{d['video_url']}", "", f"使用注意：{d['usage_caution_zh']}", "", "### 可核查的教学说明", ""])
        for n in d["teaching_notes"]:
            lines.append(f"- 发言 {n['utterance_range'][0]}—{n['utterance_range'][1]}：目标＝{n['goal_zh']}；技巧＝{n['technique_zh']}；简短依据＝{n['rationale_summary_zh']}")
        lines.extend(["", "### 完整英文原文", ""])
        for m in d["messages"]:
            role = "咨询师" if m["interlocutor"] == "therapist" else "来访者"
            lines.extend([f"**{m['utterance_id']} · {role} · {m['timestamp']}**", "", m["content"], ""])
    (ROOT / "阅读说明与完整对话.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"dialogues": len(result), "utterances": sum(d["utterance_count"] for d in result), "counts": {d["transcript_id"]: d["utterance_count"] for d in result}, "validated": "contiguous unique utterance IDs; all source utterances retained; full-version text matches"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
