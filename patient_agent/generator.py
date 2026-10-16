"""PA3 患者生成器：决定「这一轮患者该怎么回应」。

输入是病例 + 上一轮系统回复；输出是一个 ``PatientAction``——说一句话、确认画像、
提交量表题、或暂停/恢复。运行器（PA5）只负责把动作翻译成一次 API 调用，
生成器通过可替换模型客户端生成，离线测试不碰网络。

两种运行方式共用同一段 prompt 组装逻辑：
- 离线：``RuleBasedLLM`` 回显 prompt 里的 ``<output>`` 块，确定性、可复现；
- 在线：真实模型返回分段文本及依据绑定，再由独立请求复核。

生成器自己先渲染一句「参考台词」放进 ``<output>``。所以在线模式下模型是在
受约束的候选上改写；是否新增病例外事实还需要守卫验证。
"""

from __future__ import annotations

import random
import re
import json
import time
from datetime import datetime, timezone
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.client.base import ScaleItemPrompt
from patient_agent.errors import LLMError
from patient_agent.lexicon import DIAGNOSIS_LABELS
from patient_agent.llm.base import LLMClient
from patient_agent.schemas import PatientScenario, RevealableFact
from patient_agent.evidence import AnswerSegment, FactEvidence, GeneratedAnswer, PROMPT_VERSION, evidence_bundle

ActionKind = Literal["say", "confirm_profile", "submit_scale", "set_pause", "stop"]

#: 分句符。与 Mock 侧的切分保持一致（顿号不算分句），避免两边判断不一致。
_CLAUSE_RE = re.compile(r"[，。！？；\n]")


def _clauses(text: str) -> list[str]:
    return [item.strip() for item in _CLAUSE_RE.split(text) if item.strip()]


def _already_said(needle: str, haystack: str) -> bool:
    """开场是否已经表达过这句主诉，避免把同一句贴两遍。

    整句包含算说过；否则按分句比较，只要有半数以上分句已经出现就认为说过了——
    危机病例的 ``risk.utterance`` 与 ``chief_complaint`` 措辞接近但不等，
    整句匹配会判定为「没说」，从而在开场里重复一遍「想死」。
    """

    if not needle:
        return True
    if needle in haystack:
        return True
    parts = _clauses(needle)
    if not parts:
        return True
    hits = sum(1 for part in parts if part in haystack)
    return hits * 2 >= len(parts)


class PatientAction(BaseModel):
    """生成器对「这一轮做什么」的唯一输出。"""

    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    text: str = ""
    reason: str = ""
    # 这句话对应哪条病例事实；守卫据此核对「说出口的内容有没有病例依据」。
    fact_id: str = ""
    is_denial: bool = False
    # 生成这一轮时真正用过的 prompt（开发者端可见，便于排查）。
    prompt_user: str = ""
    prompt_system: str = ""
    segments: list[AnswerSegment] = Field(default_factory=list)
    authorized_evidence: list[FactEvidence] = Field(default_factory=list)
    generation_error: str = ""
    raw_output: str = ""
    generation_mode: str = "legacy"
    prompt_version: str = PROMPT_VERSION
    generated_at: str = ""
    generation_elapsed_ms: float = 0.0

    # confirm_profile
    decision: str = ""
    notes: str = ""
    # submit_scale
    item_number: str = ""
    score: int = 0
    answer_text: str = ""
    # set_pause
    paused: bool = False


class TurnContext(BaseModel):
    """运行器在每轮开始前告诉生成器的「系统侧现状」。"""

    model_config = ConfigDict(extra="ignore")

    turn_index: int = 0
    treatment_turn: int = 0
    last_reply: str = ""
    # 最近若干轮 (医生, 患者) 对话，供润色阶段承接上文；由运行器切片注入。
    recent_turns: list[tuple[str, str]] = Field(default_factory=list)
    waiting_type: str = "none"
    main_layer: str = "L1"
    scale_item: ScaleItemPrompt | None = None
    # 截至目前系统发出的全部事件（累积），用于事实解锁闸门。
    events: list[str] = Field(default_factory=list)
    session_ended: bool = False
    termination_reason: str | None = None


class PatientState(BaseModel):
    """一次疗程中患者侧的进度。``SessionRunner`` 每次运行前 ``reset``。"""

    disclosed: list[str] = Field(default_factory=list)
    seen_events: list[str] = Field(default_factory=list)
    scale_answered: dict[str, int] = Field(default_factory=dict)
    treatment_turns: int = 0
    trajectory_index: int = 0
    send_turns: int = 0
    pause_used: bool = False
    profile_asked: int = 0

    def has_disclosed(self, fact_id: str) -> bool:
        return fact_id in self.disclosed

    def mark(self, fact_id: str) -> None:
        if fact_id and fact_id not in self.disclosed:
            self.disclosed.append(fact_id)


# 分值 → 口语化作答。各量表问的「口径」不同，措辞必须跟着换：
# PHQ-9/GAD-7 问「过去两周有多少天」——频次；ISI/PCL-5/Y-BOCS 问「程度有多重」——严重度；
# MDQ 在本项目里按是否题用（见 mock_system.SCALE_MAX_SCORE 的注释）。
# 若全都套频次措辞，「入睡困难的严重程度」会被答成「一半以上的天数」这种答非所问。
_FREQUENCY_PHRASES: tuple[str, ...] = ("完全没有", "有几天", "一半以上的天数", "几乎每天")
_SEVERITY_PHRASES: tuple[str, ...] = ("没有", "轻度", "中度", "重度", "极重度")
_SATISFACTION_PHRASES: tuple[str, ...] = ("非常满意", "比较满意", "一般", "比较不满意", "非常不满意")
_YES_NO_PHRASES: tuple[str, ...] = ("否", "是")

#: 量表 → 分值锚点。表里没有的量表退回「N分」，不编造措辞。
_SCALE_PHRASES: dict[str, tuple[str, ...]] = {
    "PHQ-9": _FREQUENCY_PHRASES,
    "GAD-7": _FREQUENCY_PHRASES,
    "ISI": _SEVERITY_PHRASES,
    "PCL-5": _SEVERITY_PHRASES,
    "Y-BOCS": _SEVERITY_PHRASES,
    "MDQ": _YES_NO_PHRASES,
}

#: 同一量表内锚点方向相反的题目：ISI 第 4 题问「满意程度」，分越高越不满意，
#: 与其他题「分越高越重」相反，不能共用严重度措辞。
_SCALE_ITEM_PHRASES: dict[tuple[str, str], tuple[str, ...]] = {
    ("ISI", "4"): _SATISFACTION_PHRASES,
}


def _score_phrase(scale_name: str, item_number: str, score: int) -> str:
    """把分值译成符合该量表口径的口语作答，超出锚点表时退回「N分」。"""

    anchors = _SCALE_ITEM_PHRASES.get((scale_name, item_number)) or _SCALE_PHRASES.get(
        scale_name
    )
    if anchors is not None and 0 <= score < len(anchors):
        return anchors[score]
    return f"{score}分"


# 上下文净化：摘要与历史会进入 prompt，模型可能照抄。诊断名与危机词一旦被抄进
# 患者台词，会命中 guard 的阻断性检查（DIAGNOSIS_OVERREACH / NEVER_DISCLOSE），
# 而守卫是词表匹配、不看意图。这里做与守卫同源的替换，宁可上下文读起来别扭，
# 也不能给模型留下可抄的触发词。
_DIAGNOSIS_MASK = "（医生提过的说法）"
_RISK_MASK = "（那类念头）"


def _sanitize_context(text: str, never_disclose: Iterable[str]) -> str:
    if not text:
        return text
    for label in DIAGNOSIS_LABELS:
        if label in text:
            text = text.replace(label, _DIAGNOSIS_MASK)
    for term in never_disclose:
        if term and term in text:
            text = text.replace(term, _RISK_MASK)
    return text


def _contains_diagnosis_label(text: str) -> bool:
    return any(label in text for label in DIAGNOSIS_LABELS)


class PatientGenerator:
    """按病例脚本驱动患者行为的确定性生成器（LLM 只负责措辞）。"""

    def __init__(
        self,
        scenario: PatientScenario,
        llm: LLMClient,
        *,
        max_retries: int = 2,
        structured_output: bool = False,
    ) -> None:
        self.scenario = scenario
        self.llm = llm
        self.max_retries = max_retries
        self.structured_output = structured_output
        self.rng = random.Random(scenario.seed)
        self.state = PatientState()
        # 当前这一轮的上下文；打开场时为空。润色用的摘要与历史都从这里取。
        self._context: TurnContext | None = None
        self._facts: list[RevealableFact] = scenario.revealable_facts()
        self._fact_index: dict[str, RevealableFact] = {
            fact.fact_id: fact for fact in self._facts
        }

    # ------------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        """重跑同一病例时回到初始状态（种子也重置，保证结果一致）。"""

        self.rng = random.Random(self.scenario.seed)
        self.state = PatientState()
        self._context = None

    # ------------------------------------------------------------------ 开场
    def opening(self) -> PatientAction:
        self._context = None
        template = self.scenario.disclosure.opening_templates[0]
        # 开场句里带上主诉，给 L1 一次识别风险的机会；已经说过就不再重复。
        complaint = self.scenario.clinical.chief_complaint
        text = template if _already_said(complaint, template) else f"{template}{complaint}"
        return self._say(text, fact_id="opening", reason="开场陈述")

    # ------------------------------------------------------------ 主分发
    def next_action(self, context: TurnContext) -> PatientAction:
        self._context = context
        self._absorb_events(context.events)

        if context.session_ended:
            return PatientAction(kind="stop", reason=f"会话已结束：{context.termination_reason}")

        if context.waiting_type == "pause":
            if self.scenario.pause_plan.resume_after_pause:
                return PatientAction(
                    kind="set_pause",
                    paused=False,
                    reason="按计划从暂停中恢复",
                    notes=self.scenario.pause_plan.resume_note,
                )
            return PatientAction(kind="stop", reason="按计划保持暂停")

        if context.waiting_type == "profile_confirmation":
            return self._confirm_action()

        if context.waiting_type == "scale_item" and context.scale_item is not None:
            return self._scale_action(context.scale_item)

        if context.main_layer == "L4" or context.treatment_turn > 0:
            return self._treatment_action(context)

        return self._interview_action(context)

    # ------------------------------------------------------------ 访谈阶段
    def _interview_action(self, context: TurnContext) -> PatientAction:
        fact = self._pick_fact(context.last_reply)
        if fact is not None:
            return self._fact_action(fact, context)
        # 没有被追问到的新内容：按行为参数决定敷衍还是反问。
        if self._roll(self.scenario.behavior.ask_back):
            return self._say("你觉得呢？我现在也说不太清楚。", reason="患者反问")
        return self._say(self._filler(), reason="本轮无新事实")

    # ------------------------------------------------------------ 治疗阶段
    def _treatment_action(self, context: TurnContext) -> PatientAction:
        # 先取下标再推进：``_next_trajectory_step`` 成功时会把指针前移，
        # 用移位后的值当 fact_id 会让标签和实际说的那一步错开一位。
        index = self.state.trajectory_index
        step = self._next_trajectory_step(context.treatment_turn)
        if step is not None:
            return self._say(
                step.utterance,
                reason=f"轨迹推进：{step.kind}",
                fact_id=f"trajectory:{index}",
            )
        fact = self._pick_fact(context.last_reply)
        if fact is not None:
            return self._fact_action(fact, context)
        return self._say(self._filler(), reason="治疗期本轮无新事实")

    def _next_trajectory_step(self, treatment_turn: int):
        steps = self.scenario.trajectory.steps
        index = self.state.trajectory_index
        if index >= len(steps):
            return None
        step = steps[index]
        if step.after_l4_turn > treatment_turn:
            return None
        self.state.trajectory_index += 1
        return step

    # ------------------------------------------------------------ 事实选择
    def _unlocked(self, fact: RevealableFact) -> bool:
        if not fact.unlock_after_events:
            return True
        seen = set(self.state.seen_events)
        return any(event in seen for event in fact.unlock_after_events)

    def _pick_fact(self, probe_text: str) -> RevealableFact | None:
        """挑一条本轮该说的事实。

        优先级：被追问命中的 > 主动披露的 > 什么都不说。
        否认类事实（denial_expected）**必须**被追问才说，否则会出现「没被问就说没有」
        这种不自然的对话，也会让「不迎合」验证失去意义。
        """

        candidates = [
            fact
            for fact in self._facts
            if not self.state.has_disclosed(fact.fact_id) and self._unlocked(fact)
        ]
        if not candidates:
            return None

        scored: list[tuple[int, int, RevealableFact]] = []
        for fact in candidates:
            hits = sum(1 for keyword in fact.probe_keywords if keyword and keyword in probe_text)
            if hits:
                scored.append((hits, -fact.priority, fact))
        if scored:
            # 命中越多越优先；同命中数取 priority 更小（更重要）的。
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return scored[0][2]

        proactive = [
            fact
            for fact in candidates
            if fact.disclosure == "proactive" and not fact.denial_expected
        ]
        if proactive:
            return min(proactive, key=lambda fact: fact.priority)
        return None

    def _fact_action(self, fact: RevealableFact, context: TurnContext) -> PatientAction:
        self.state.mark(fact.fact_id)
        return self._render_fact(fact)

    def _render_fact(self, fact: RevealableFact, *, note: str = "") -> PatientAction:
        if fact.denial_expected:
            return self._say(
                fact.denial_utterance,
                fact_id=fact.fact_id,
                reason=f"否认事实：{fact.fact_id}{note}",
                is_denial=True,
            )
        return self._say(
            fact.utterance,
            fact_id=fact.fact_id,
            reason=f"披露事实：{fact.fact_id}{note}",
        )

    def retry_action(self, action: PatientAction, *, feedback: str = "") -> PatientAction:
        self._retry_feedback = feedback
        try:
            return self._retry_action(action)
        finally:
            self._retry_feedback = ""

    def _retry_action(self, action: PatientAction) -> PatientAction:
        """守卫判定违规后重说一遍。

        用的是**同一条事实**，只重新措辞（在线模式下就是重新采样一次）——
        如果改成重新选题，会把本来该说的事实丢掉，掩盖真实问题。
        """

        fact = self._fact_index.get(action.fact_id) if action.fact_id else None
        if fact is not None:
            return self._render_fact(fact, note="（重说）")
        return self._say(
            self._reference_line(action.prompt_user) or "嗯，我在听。",
            fact_id=action.fact_id,
            reason=f"{action.reason}（重说）",
            is_denial=action.is_denial,
        )

    # ------------------------------------------------------------ 画像确认
    def _confirm_action(self) -> PatientAction:
        plan = self.scenario.profile_confirmation
        decision = plan.decision_for(self.state.profile_asked)
        self.state.profile_asked += 1
        return PatientAction(
            kind="confirm_profile",
            decision=decision,
            notes=plan.notes,
            reason=f"画像确认：{decision}",
        )

    # ------------------------------------------------------------ 量表作答
    def _scale_action(self, item: ScaleItemPrompt) -> PatientAction:
        plan = next(
            (
                candidate
                for candidate in self.scenario.clinical.scale_plans
                if candidate.scale_name == item.scale_name
            ),
            None,
        )
        answered = self.state.scale_answered.get(item.scale_name, 0)

        if plan is None:
            # 系统问了一个病例没计划的量表：给最低分，避免编造。
            return self._submit(item, 0)

        if plan.willingness == "refuse":
            return self._refuse_scale(plan.comment)

        if plan.refuse_after_items is not None and answered >= plan.refuse_after_items:
            return self._refuse_scale(plan.comment)

        score = plan.answers[answered] if answered < len(plan.answers) else 0
        clamp = max(item.min_score, min(item.max_score, score))
        return self._submit(item, clamp)

    def _refuse_scale(self, comment: str = "") -> PatientAction:
        refusals = self.scenario.disclosure.refusal_utterances or ["这个我不太想做。"]
        text = refusals[0] + (f"（{comment}）" if comment else "")
        return self._say(text, reason="拒绝量表")

    def _submit(self, item: ScaleItemPrompt, score: int) -> PatientAction:
        self.state.scale_answered[item.scale_name] = (
            self.state.scale_answered.get(item.scale_name, 0) + 1
        )
        phrase = _score_phrase(item.scale_name, item.item_number, score)
        return PatientAction(
            kind="submit_scale",
            item_number=item.item_number,
            score=score,
            answer_text=phrase,
            reason=f"量表 {item.scale_name} 第 {item.item_number} 题答 {score} 分",
        )

    # ------------------------------------------------------------ 文本渲染
    def _roll(self, probability: float) -> bool:
        return self.rng.random() < probability

    def _filler(self) -> str:
        replies = self.scenario.disclosure.non_fact_replies
        if not replies:
            return "嗯，我在听。"
        index = self.rng.randrange(len(replies))
        return replies[index]

    def _say(
        self,
        text: str,
        *,
        reason: str = "",
        fact_id: str = "",
        is_denial: bool = False,
    ) -> PatientAction:
        system_prompt, user_prompt = self._build_prompt(text, fact_id, is_denial)
        bundle = evidence_bundle(self.scenario, fact_id)
        structured = self.structured_output
        started = time.perf_counter()
        generated_at = datetime.now(timezone.utc).isoformat()
        error = ""
        segments = []
        if structured:
            user_prompt += "\n本轮唯一授权依据（医生问题和历史不是新增事实来源）：\n" + json.dumps([e.model_dump(mode="json") for e in bundle], ensure_ascii=False)
            user_prompt += '\n仅返回JSON：{"segments":[{"segment_id":"s1","text":"患者原话","kind":"factual","fact_ids":["本轮依据编号"]}]}。非事实片段用non_factual及空fact_ids。禁止返回额外最终文本；不得承认未记录经历。'
            try:
                raw = self.llm.complete(system=system_prompt, user=user_prompt, temperature=0.3, max_tokens=1200).strip()
                parsed = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw) if raw.startswith("```") and raw.endswith("```") else raw
                segments = GeneratedAnswer.model_validate_json(parsed).segments
                rendered = "".join(s.text for s in segments)
            except Exception as exc:
                error = type(exc).__name__
                raw = locals().get("raw", "")
                rendered = ""
        else:
            raw = rendered = self._invoke_llm(system_prompt, user_prompt)
            # 参考台词回显可确定绑定；任意改写不能自动伪造依据。
            match = next((e for e in bundle if e.value == rendered), None)
            if match:
                segments = [AnswerSegment(segment_id="s1", text=rendered,
                    kind="non_factual" if match.status == "non_factual" else "factual",
                    fact_ids=[] if match.status == "non_factual" else [match.fact_id])]
        return PatientAction(
            kind="say",
            text=rendered,
            reason=reason,
            fact_id=fact_id,
            is_denial=is_denial,
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            segments=segments, authorized_evidence=bundle, generation_error=error,
            raw_output=raw, generation_mode="structured" if structured else "legacy",
            generated_at=generated_at, generation_elapsed_ms=(time.perf_counter()-started)*1000,
        )

    def _invoke_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调一次 LLM；离线适配器回显 ``<output>``，在线适配器自由改写。

        LLM 不可用或连续失败时退回参考台词——「病例脚本」才是唯一事实来源，
        措辞失败不该让整条会话崩掉。
        """

        last_error: Exception | None = None
        for _ in range(max(1, self.max_retries)):
            try:
                text = self.llm.complete(system=system_prompt, user=user_prompt)
                if text.strip():
                    return text.strip()
            except LLMError as exc:  # pragma: no cover - 仅在线路径触发
                last_error = exc
        fallback = self._reference_line(user_prompt)
        if last_error is not None:  # pragma: no cover
            return fallback
        return fallback

    @staticmethod
    def _reference_line(user_prompt: str) -> str:
        """从 prompt 的 ``<output>`` 块取回参考台词；解析失败时返回空串。"""

        start = user_prompt.find("<output>")
        end = user_prompt.find("</output>", start + 1)
        if start == -1 or end == -1:
            return ""
        return user_prompt[start + len("<output>") : end].strip()

    # ─────────────────────────────────────────────────────────────────────────
    # 【接入点 14】患者人设与边界的 prompt（真模型下最该盯的地方）
    # 接入什么：真实大模型的 system/user prompt。接上真模型后，患者说得像不像、
    #           会不会露馅（说出内部术语、自己下诊断），都由这段文本决定。
    # 何时需要改：
    #   * 真实模型爱说内部词 / 爱总结诊断 → 加强下面的禁止事项；
    #   * 病例的 utterance 模板想换风格 → 改「参考台词」段；
    #   * 离线与在线必须共用这一份文本，改完两边同时生效（这正是可复现的前提）。
    # 注意：prompt 里绝不能塞病例以外的临床事实，否则「不添加病例外事实」的验收会失守。
    # ─────────────────────────────────────────────────────────────────────────
    def _build_prompt(
        self, reference: str, fact_id: str, is_denial: bool
    ) -> tuple[str, str]:
        """组装 system/user prompt。

        结构固定为「角色设定 → 病例事实 → 禁止事项 → 参考台词」。离线与在线
        共用同一份文本，唯一区别是适配器怎么用它。
        """

        scenario = self.scenario
        behavior = scenario.behavior
        style = {
            "terse": "每次只说一两句，话不多",
            "medium": "每次说两三句，像日常聊天",
            "verbose": "每次说三四句，会主动补充细节",
        }.get(behavior.verbosity, "每次说两三句")

        system_lines = [
            "你在扮演一位前来就诊的来访者（患者），不是医生，也不是助手。",
            f"你的年龄大致在 {scenario.background.age_range}，职业是 {scenario.background.occupation}。",
            f"表达风格：{style}；情绪强度约 {behavior.emotional_intensity}/10。",
            "你只知道自己经历的事，不知道任何诊断结论、量表得分标准或系统内部结构。",
            "绝对不要说出「系统提示」「prompt」「L1~L5」「置信度」「阈值」这类内部词汇。",
            "不要自己下诊断结论，最多转述别人对你说过的话。",
        ]
        if scenario.disclosure.never_disclose:
            system_lines.append(
                "以下内容绝不能主动提起：" + "、".join(scenario.disclosure.never_disclose) + "。"
            )
        summary = self._render_case_summary()
        if summary:
            system_lines += [
                "",
                "## 我的基本情况（用来保持前后一致，不要逐条复述）",
                summary,
            ]

        history = self._render_history()
        fact_line = self._fact_description(fact_id, is_denial)
        user_lines = ["## 这一轮要传达的经历", fact_line, ""]
        if history:
            user_lines += [
                "## 最近的对话（用一句自然的话回应医生最后说的那句）",
                history,
                "",
            ]
        user_lines += [
            "## 说话要求",
            "用口语第一人称回答，不要解释你在扮演角色，不要重复对方的问句。",
            "只能使用本轮授权依据中明确出现的数量、单位和时间范围；不能估算或借用其他字段的数字。",
        ]
        if is_denial:
            user_lines.append("这一轮是在诚实地否认，语气要自然，不要显得像在背诵。")
        user_lines += [
            "",
            "## 参考台词（内容要点以此为准，可以自然改写）",
            f"<output>{reference}</output>",
        ]
        if getattr(self, "_retry_feedback", ""):
            user_lines += ["上次回答未通过事实守卫；只重述参考台词，不承认医生暗示的未知经历。",
                           self._retry_feedback]
        return "\n".join(system_lines), "\n".join(user_lines)

    def _render_case_summary(self) -> str:
        """把病例里「患者本可以说出口」的部分拼成摘要，供模型保持前后一致。

        白名单渲染：只放主诉、症状标签、咨询目标与矛盾心理。刻意排除诊断编码、
        严重度、病程数字、量表得分与风险等级——前几项会触发守卫的阻断性检查，
        诊断编码则属于患者不该知道的判读结论。
        """

        scenario = self.scenario
        lines = [f"主诉：{scenario.clinical.chief_complaint}"]
        labels = [
            symptom.label
            for symptom in scenario.clinical.symptoms
            if symptom.present and not _contains_diagnosis_label(symptom.label)
        ]
        if labels:
            lines.append("困扰我的方面：" + "、".join(labels))
        if scenario.psychology.goal:
            lines.append(f"我希望：{scenario.psychology.goal}")
        if scenario.psychology.ambivalence:
            lines.append(f"我的顾虑：{scenario.psychology.ambivalence}")
        return _sanitize_context("\n".join(lines), scenario.disclosure.never_disclose)

    def _render_history(self) -> str:
        """渲染注入的最近若干轮医患对话；医生的话里可能带诊断名，同样要净化。"""

        turns = self._context.recent_turns if self._context else []
        lines: list[str] = []
        for doctor, patient in turns:
            if doctor:
                lines.append(f"医生：{doctor}")
            if patient:
                lines.append(f"我：{patient}")
        return _sanitize_context("\n".join(lines), self.scenario.disclosure.never_disclose)

    def _fact_description(self, fact_id: str, is_denial: bool) -> str:
        if not fact_id:
            return "（本轮没有必须传达的新信息，可以回应对方的问题或表示还没想清楚。）"
        fact = self._fact_index.get(fact_id)
        if fact is None:
            return f"（{fact_id}）"
        if is_denial:
            return f"（否认或澄清：{fact_id}）{fact.denial_utterance}"
        return f"（{fact.kind} / {fact_id}）{fact.utterance}"

    # ------------------------------------------------------------ 事件吸收
    def _absorb_events(self, events: list[str]) -> None:
        for event in events:
            if event not in self.state.seen_events:
                self.state.seen_events.append(event)

    # ------------------------------------------------------------ 给运行器用
    def action_to_send_text(self, action: PatientAction) -> str:
        """把动作转成「患者说出口的话」，用于日志与守卫。"""

        if action.kind == "say":
            return action.text
        if action.kind == "submit_scale":
            return action.answer_text or str(action.score)
        if action.kind == "confirm_profile":
            return f"[画像确认：{action.decision}]"
        if action.kind == "set_pause":
            return "[暂停]" if action.paused else "[恢复]"
        return "[结束]"
