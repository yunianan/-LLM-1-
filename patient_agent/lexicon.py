"""共享词汇与文本抽取工具。

只放「与病例无关的通用词表」：风险词、否认词、严重度词、内部术语、诊断标签，
以及数字/时长抽取。病例自身的关键词一律写在病例 JSON 里（probe_keywords / denial_utterance），
不在这里硬编码，避免把临床判断写进代码。
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 12】风险 / 否认词表（与真实系统的措辞对齐）
# 接入什么：本文件是 Patient Agent 侧的通用词表。接上真实系统后，如果真实 L1/L3 识别
#           用的措辞与这里不同，需要按真实系统的实际用词校准，否则守卫/评测会误判。
# 边界：只放与病例无关的通用词；病例自身的关键词写在病例 JSON 里
#       （probe_keywords / denial_utterance），不要把临床判断写进代码。
# ─────────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------- 风险
# 从低到高；guard 用顺序判断「患者是否把风险说得比病例脚本更重」。
RISK_TIER_ORDER: dict[str, int] = {
    "none": 0,
    "ambiguous": 1,
    "explicit_ideation": 2,
    "plan": 3,
    "immediate_crisis": 4,
}

RISK_TERMS: dict[str, tuple[str, ...]] = {
    "immediate_crisis": ("现在就去", "马上就做", "已经准备好了", "正在准备", "就在手边"),
    "plan": ("计划好了", "想好怎么做", "写好了", "买了药", "准备好了方法"),
    "explicit_ideation": ("想死", "不想活", "自杀", "结束生命", "一了百了", "活不下去", "死了算了", "伤害自己"),
    "ambiguous": ("活着没意思", "活着真没什么意思", "活着没什么意思", "没有意义", "消失", "活着很累", "不如死", "撑不下去"),
}

# 患者「否认」的措辞。否认句里出现风险词是正常的（例如「我没有想过自杀」），
# 守卫据此把否认句与真正的矛盾表述区分开。
DENIAL_TERMS: tuple[str, ...] = (
    "没有",
    "没想过",
    "不会那样",
    "从来没有",
    "没这个念头",
    "我不会",
    "不怎么想",
)

# ------------------------------------------------------------------- 严重度
SEVERITY_HIGH_TERMS: tuple[str, ...] = ("每天", "时刻", "一直", "完全", "极度", "非常严重", "都")
SEVERITY_LOW_TERMS: tuple[str, ...] = ("偶尔", "有时", "一两次", "还可以", "不太")

# --------------------------------------------------- 内部术语（患者不应知道）
# 患者说出这些词说明生成器发生了「系统提示/内部结构泄露」。
INTERNAL_JARGON: tuple[str, ...] = (
    "系统提示",
    "提示词",
    "prompt",
    "Prompt",
    "schema",
    "Schema",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "编排",
    "编排器",
    "置信度",
    "阈值",
    "mock",
    "Mock",
    "测试用例",
    "种子",
    "seed",
    "CLINICR",
    "Verifier",
    "RAG",
    "知识库",
    "OARS",
    "MI阶段",
    "量表得分规则",
    "证据节点",
)

# ------------------------------------------------- 诊断标签（患者不得自行下结论）
# 患者可以说「医生说我抑郁」（走病例的 patient_reported_labels 白名单），
# 但不能自己冒出新诊断名，否则既越界也让「不迎合」验证失效。
DIAGNOSIS_LABELS: tuple[str, ...] = (
    "重度抑郁症",
    "中度抑郁症",
    "抑郁症",
    "抑郁障碍",
    "双相障碍",
    "双相情感障碍",
    "躁狂发作",
    "精神分裂症",
    "广泛性焦虑障碍",
    "惊恐障碍",
    "创伤后应激障碍",
    "强迫症",
    "社交焦虑障碍",
    "注意缺陷多动障碍",
)

# ------------------------------------------------------------------- 数字抽取
NUMBER_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?(?:~\d+)?)\s*(?:个多|多|个)?\s*(星期|礼拜|月|周|天|年|小时|分钟|岁|片|颗|次|晚|点|分)"
)
PLAIN_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_CHINESE_QUANTITY = re.compile(r"[零〇一二两三四五六七八九十百千]+(?=(?:个多|多|个)?(?:月|星期|礼拜|周|天|年|小时|分钟|岁|片|颗|次|晚|点|分))")
_DIGITS = {char: value for value, chars in enumerate(("零〇", "一", "二两", "三", "四", "五", "六", "七", "八", "九")) for char in chars}


def normalize_quantities(text: str) -> str:
    """统一中文/阿拉伯数字；保留「三四」这类范围，不把它误读成三十四。"""

    def replace(match: re.Match[str]) -> str:
        word = match.group()
        if all(char in _DIGITS for char in word):
            digits = [_DIGITS[char] for char in word]
            if len(digits) == 2 and digits[1] == digits[0] + 1:
                return f"{digits[0]}~{digits[1]}"
            return "".join(str(value) for value in digits)
        total = current = 0
        for char in word:
            if char in _DIGITS:
                current = _DIGITS[char]
            else:
                total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
                current = 0
        return str(total + current)

    return _CHINESE_QUANTITY.sub(replace, text)


def is_negated_occurrence(text: str, index: int) -> bool:
    """只识别紧邻该风险词的否认，其他分句的「没有」不能掩盖风险。"""

    prefix = re.split(r"[，。！？；\n]|但是|可是|不过|然而|但", text[:index])[-1]
    return bool(re.search(
        r"(?:没有|没|不曾|不会|并无|不)(?:想过|想到|想|过|出现过|出现|产生过|产生|发生过|发生|任何|这种|那种|这样的|要|打算|再|去|做|的)*[‘\"“]?$",
        prefix,
    ))


def asserted_risk_tier(text: str) -> str:
    """取未被局部否认的最高风险级别，不信任 action.is_denial 标志。"""

    best = "none"
    for tier, terms in RISK_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                if not is_negated_occurrence(text, match.start()) and RISK_TIER_ORDER[tier] > RISK_TIER_ORDER[best]:
                    best = tier
    return best


def extract_number_tokens(text: str) -> set[str]:
    """抽出文本里的数字标记。

    同时给出「数字+单位」和「裸数字」两种形式，守卫用病例声明的集合做比对。
    这是「不添加病例外事实」的近似检查：患者随口编造的时长/次数会在裸数字上露馅。
    """

    tokens: set[str] = set()
    normalized = normalize_quantities(text)
    for match in NUMBER_UNIT_RE.finditer(normalized):
        unit = {"星期": "周", "礼拜": "周"}.get(match.group(2), match.group(2))
        tokens.add(match.group(1) + unit)
    for match in PLAIN_NUMBER_RE.finditer(normalized):
        tokens.add(match.group(0))
    return tokens


def detect_risk_tier(text: str) -> str:
    """返回文本中出现的最高风险层级；没有任何风险词时返回 "none"。"""

    best = "none"
    for tier, terms in RISK_TERMS.items():
        if any(term in text for term in terms):
            if RISK_TIER_ORDER[tier] > RISK_TIER_ORDER[best]:
                best = tier
    return best


def has_denial(text: str) -> bool:
    return any(term in text for term in DENIAL_TERMS)


def contains_any(text: str, terms: object) -> bool:
    """terms 可以是任意可迭代的字符串集合。"""

    return any(term in text for term in terms)  # type: ignore[operator]
