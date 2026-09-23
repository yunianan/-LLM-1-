from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ScaleItemDefinition:
    prompt: str
    min_score: int
    max_score: int
    anchors: dict[int, str]


@dataclass(frozen=True)
class ScaleDefinition:
    name: str
    purpose: str
    timeframe: str
    items: tuple[ScaleItemDefinition, ...]
    threshold: Callable[[dict[int, int]], str]

    def total_score(self, answers: dict[int, int]) -> int:
        if self.name == "MDQ":
            return sum(answers.get(number, 0) for number in range(1, 14))
        return sum(answers.values())


FREQUENCY_03 = {0: "完全没有", 1: "有几天", 2: "一半以上时间", 3: "几乎每天"}
SEVERITY_04 = {0: "没有", 1: "轻微", 2: "中等", 3: "相当严重", 4: "极其严重"}
YES_NO = {0: "否", 1: "是"}
YBOCS_TIME = {
    0: "没有占用时间",
    1: "每天少于1小时",
    2: "每天1到3小时",
    3: "每天3到8小时",
    4: "每天超过8小时",
}
YBOCS_INTERFERENCE = {
    0: "没有干扰",
    1: "轻微干扰但整体表现未受损",
    2: "明确干扰但仍能应对",
    3: "造成明显功能损害",
    4: "几乎无法正常活动",
}
YBOCS_DISTRESS = {
    0: "没有痛苦",
    1: "轻微且不太困扰",
    2: "明显困扰但仍能应对",
    3: "非常困扰",
    4: "极度且近乎持续的痛苦",
}
YBOCS_RESISTANCE = {
    0: "始终努力抵抗",
    1: "大部分时间努力抵抗",
    2: "会做一些抵抗",
    3: "通常不抵抗但仍有不情愿",
    4: "完全不抵抗",
}
YBOCS_CONTROL = {
    0: "完全能够控制",
    1: "大部分能够控制",
    2: "有一定控制",
    3: "很少能够控制",
    4: "完全无法控制",
}
ISI_SYMPTOM = {0: "没有", 1: "轻度", 2: "中度", 3: "重度", 4: "极重度"}
ISI_SATISFACTION = {
    0: "非常满意",
    1: "满意",
    2: "一般",
    3: "不满意",
    4: "非常不满意",
}
ISI_IMPACT = {0: "完全没有", 1: "一点", 2: "有些", 3: "很多", 4: "非常多"}


def _phq_threshold(items: dict[int, int]) -> str:
    total = sum(items.values())
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately_severe"
    return "severe"


def _gad_threshold(items: dict[int, int]) -> str:
    total = sum(items.values())
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def _pcl_threshold(items: dict[int, int]) -> str:
    return "positive_screen" if sum(items.values()) >= 31 else "subthreshold"


def _ybocs_threshold(items: dict[int, int]) -> str:
    total = sum(items.values())
    if total <= 7:
        return "subclinical"
    if total <= 15:
        return "mild"
    if total <= 23:
        return "moderate"
    if total <= 31:
        return "severe"
    return "extreme"


def _isi_threshold(items: dict[int, int]) -> str:
    total = sum(items.values())
    if total <= 7:
        return "no_clinically_significant_insomnia"
    if total <= 14:
        return "subthreshold"
    if total <= 21:
        return "moderate"
    return "severe"


def _mdq_threshold(items: dict[int, int]) -> str:
    symptom_count = sum(items.get(number, 0) for number in range(1, 14))
    same_period = items.get(14) == 1
    impairment = items.get(15, 0) >= 2
    return (
        "positive_screen"
        if symptom_count >= 7 and same_period and impairment
        else "subthreshold"
    )


def _items(prompts: list[str], anchors: dict[int, str]) -> tuple[ScaleItemDefinition, ...]:
    return tuple(
        ScaleItemDefinition(prompt, min(anchors), max(anchors), anchors)
        for prompt in prompts
    )


SCALE_CATALOG: dict[str, ScaleDefinition] = {
    "PHQ-9": ScaleDefinition(
        name="PHQ-9",
        purpose="了解最近两周抑郁相关表现的频率",
        timeframe="最近两周",
        items=_items([
            "最近两周，你有多频繁对做事情提不起兴趣或感受不到乐趣？",
            "最近两周，你有多频繁感到情绪低落、沮丧或没有希望？",
            "最近两周，你有多频繁出现入睡困难、睡不安稳或睡得过多？",
            "最近两周，你有多频繁感到疲倦或精力不足？",
            "最近两周，你有多频繁出现食欲不佳或吃得过多？",
            "最近两周，你有多频繁觉得自己很糟、失败或让自己或家人失望？",
            "最近两周，你有多频繁难以集中注意力？",
            "最近两周，你有多频繁动作或说话明显变慢，或烦躁得坐立不安？",
            "最近两周，你有多频繁出现不如死去或伤害自己的想法？",
        ], FREQUENCY_03),
        threshold=_phq_threshold,
    ),
    "GAD-7": ScaleDefinition(
        name="GAD-7",
        purpose="了解最近两周焦虑相关表现的频率",
        timeframe="最近两周",
        items=_items([
            "最近两周，你有多频繁感到紧张、焦虑或急切？",
            "最近两周，你有多频繁无法停止或控制担忧？",
            "最近两周，你有多频繁对各种事情担忧过多？",
            "最近两周，你有多频繁难以放松？",
            "最近两周，你有多频繁因不安而难以静坐？",
            "最近两周，你有多频繁变得容易烦恼或急躁？",
            "最近两周，你有多频繁感到好像会发生可怕的事？",
        ], FREQUENCY_03),
        threshold=_gad_threshold,
    ),
    "PCL-5": ScaleDefinition(
        name="PCL-5",
        purpose="了解过去一个月创伤事件相关困扰程度",
        timeframe="过去一个月",
        items=_items([
            "不想要的、令人不安的事件记忆反复出现，对你造成多大困扰？",
            "与事件有关的噩梦对你造成多大困扰？",
            "仿佛事件再次发生的体验对你造成多大困扰？",
            "被提醒事件时强烈难过，对你造成多大困扰？",
            "被提醒事件时出现明显身体反应，对你造成多大困扰？",
            "回避与事件有关的记忆、想法或感受，对你造成多大困扰？",
            "回避让你想起事件的人、地点或活动，对你造成多大困扰？",
            "记不起事件的重要部分，对你造成多大困扰？",
            "对自己、他人或世界持续抱有负面看法，对你造成多大困扰？",
            "因事件责怪自己或他人，对你造成多大困扰？",
            "持续出现恐惧、内疚、愤怒等强烈负面情绪，对你造成多大困扰？",
            "对曾喜欢的活动失去兴趣，对你造成多大困扰？",
            "感到与他人疏远，对你造成多大困扰？",
            "难以体验积极情绪，对你造成多大困扰？",
            "容易激怒或出现攻击行为，对你造成多大困扰？",
            "冒险或可能伤害自己的行为，对你造成多大困扰？",
            "过度警觉，对你造成多大困扰？",
            "容易受惊，对你造成多大困扰？",
            "难以集中注意力，对你造成多大困扰？",
            "难以入睡或保持睡眠，对你造成多大困扰？",
        ], SEVERITY_04),
        threshold=_pcl_threshold,
    ),
    "Y-BOCS": ScaleDefinition(
        name="Y-BOCS",
        purpose="了解强迫想法和强迫行为的严重程度",
        timeframe="最近一周",
        items=(
            ScaleItemDefinition("强迫想法每天占用多少时间？", 0, 4, YBOCS_TIME),
            ScaleItemDefinition("强迫想法对生活造成多大干扰？", 0, 4, YBOCS_INTERFERENCE),
            ScaleItemDefinition("强迫想法带来多大痛苦？", 0, 4, YBOCS_DISTRESS),
            ScaleItemDefinition("你会在多大程度上抵抗强迫想法？", 0, 4, YBOCS_RESISTANCE),
            ScaleItemDefinition("你能多大程度控制强迫想法？", 0, 4, YBOCS_CONTROL),
            ScaleItemDefinition("强迫行为每天占用多少时间？", 0, 4, YBOCS_TIME),
            ScaleItemDefinition("强迫行为对生活造成多大干扰？", 0, 4, YBOCS_INTERFERENCE),
            ScaleItemDefinition("若不能执行强迫行为，你会有多痛苦？", 0, 4, YBOCS_DISTRESS),
            ScaleItemDefinition("你会在多大程度上抵抗强迫行为？", 0, 4, YBOCS_RESISTANCE),
            ScaleItemDefinition("你能多大程度控制强迫行为？", 0, 4, YBOCS_CONTROL),
        ),
        threshold=_ybocs_threshold,
    ),
    "ISI": ScaleDefinition(
        name="ISI",
        purpose="了解最近两周失眠问题和影响程度",
        timeframe="最近两周",
        items=(
            ScaleItemDefinition("最近两周，入睡困难对你来说有多严重？", 0, 4, ISI_SYMPTOM),
            ScaleItemDefinition("最近两周，睡眠维持困难对你来说有多严重？", 0, 4, ISI_SYMPTOM),
            ScaleItemDefinition("最近两周，过早醒来对你来说有多严重？", 0, 4, ISI_SYMPTOM),
            ScaleItemDefinition("你对目前睡眠状况有多满意？", 0, 4, ISI_SATISFACTION),
            ScaleItemDefinition("睡眠问题在多大程度上影响了白天功能？", 0, 4, ISI_IMPACT),
            ScaleItemDefinition("他人能在多大程度上看出睡眠问题影响了你？", 0, 4, ISI_IMPACT),
            ScaleItemDefinition("你对目前睡眠问题有多担心或痛苦？", 0, 4, ISI_IMPACT),
        ),
        threshold=_isi_threshold,
    ),
    "MDQ": ScaleDefinition(
        name="MDQ",
        purpose="了解既往情绪或精力异常升高的表现",
        timeframe="既往任何时期",
        items=(
            *_items([
                "是否曾有一段时间情绪特别好或兴奋，以至于他人觉得你和平时不同？",
                "是否曾有一段时间特别易怒，甚至与人争吵？",
                "是否曾有一段时间比平时更自信？",
                "是否曾明显减少睡眠却不觉得困？",
                "是否曾比平时说话更多或更快？",
                "是否曾感到想法飞快或停不下来？",
                "是否曾很容易分心？",
                "是否曾精力明显增加？",
                "是否曾比平时活跃或同时做更多事情？",
                "是否曾比平时更爱社交或外向？",
                "是否曾对性比平时更感兴趣？",
                "是否曾做出不符合平时习惯、可能惹麻烦的事情？",
                "是否曾出现不理性的消费或财务行为？",
            ], YES_NO),
            ScaleItemDefinition("上述表现是否有多项发生在同一时期？", 0, 1, YES_NO),
            ScaleItemDefinition(
                "这些表现造成了多大程度的问题？",
                0,
                3,
                {0: "没有问题", 1: "轻微问题", 2: "中等问题", 3: "严重问题"},
            ),
        ),
        threshold=_mdq_threshold,
    ),
}


SCALE_DOMAIN_TO_NAME = {
    "depression": "PHQ-9",
    "anxiety": "GAD-7",
    "trauma": "PCL-5",
    "obsession_compulsion": "Y-BOCS",
    "insomnia": "ISI",
    "bipolar": "MDQ",
}
