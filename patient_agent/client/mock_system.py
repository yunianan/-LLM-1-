"""内置「简化五层状态机」，作为顶层编排器就绪前的替身。

它实现与真实系统完全相同的 8 个接口，内部按 L1→L5 的顺序做最小可用的判断：

    L1  风险关键词分级：明确意念/计划/即时危机 → 安全路径，不进后续层；
        模糊风险 → 记录但继续 L2。
    L2  MI 探索 → 量表逐题作答 → 画像确认；信息不足时接受 L3 回环并补字段。
    L3  按症状生成候选、按排除条件剔除候选（例如仅有易激惹+睡眠减少不足以支持双相），
        每条判断同时挂患者证据与临床知识引用。
    L4  治疗对话；从患者表达里读改善/无改善/退出信号，推进疗程。
    L5  每轮审核 + 终止时产出 EpisodeSummary。

刻意保持独立：它不读病例文件，只用患者说出来的话做判断，否则「自动测试」就变成
自问自答。词表也自成一套，不与 Patient Agent 侧共用。
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 4】内置五层桩（真实编排器的替身）
# 接入什么：本文件不是被接入的对象，而是「真编排器就绪前」的对照组。
# 何时更换：SYSTEM_API_MODE=http 之后本文件不参与主链路。建议保留 —— 单元测试仍然
#           用它（确定性、零延迟、不出网）。真实系统稳定上线后再决定是否删除。
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from patient_agent.diagnostic_catalog import adapt_candidates, SUPPORTED_DISORDERS

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from patient_agent.client.base import (
    DevState,
    EpisodeSummary,
    MainLayer,
    ProfileDecision,
    RouteInfo,
    ScaleItemPrompt,
    SessionHistoryItem,
    SessionSnapshot,
    SystemApiClient,
    TurnResponse,
    WaitingType,
    new_session_id,
)
from patient_agent.errors import ErrorCode

# --------------------------------------------------------------------- 词表
SYMPTOM_LEXICON: dict[str, tuple[str, ...]] = {
    "depressed_mood": ("低落", "难过", "沮丧", "情绪不好", "情绪差", "心里很堵", "堵得慌", "没希望"),
    "anhedonia": ("提不起劲", "提不起兴趣", "没兴趣", "什么都不想做", "看不下去", "不想碰"),
    "insomnia": ("失眠", "睡不着", "睡不好", "早醒", "容易醒", "睡眠差", "熬夜", "经常醒"),
    "fatigue": ("没力气", "疲惫", "很累", "没有精力", "整个人很沉", "撑着"),
    "worthlessness": ("没用", "不够好", "做不好", "让人失望", "自责"),
    "concentration_difficulty": ("无法集中", "注意力", "没法专心", "脑子很乱"),
    "anxiety": ("焦虑", "紧张", "担心", "心慌", "不安", "心里紧"),
    "trauma_related": ("创伤", "噩梦", "闪回", "惊吓"),
    "obsession_compulsion": ("强迫", "反复想", "反复做", "停不下来"),
    "elevated_mood": ("异常兴奋", "不用睡觉", "不睡不累", "花钱多", "话特别多"),
    "irritability": ("烦躁", "容易烦", "易激惹", "发脾气", "烦得慌", "发火", "上火", "就烦"),
}

EXCLUSION_LEXICON: dict[str, tuple[str, ...]] = {
    "manic_history": ("异常兴奋", "不用睡觉", "不睡不累", "花钱多", "话特别多", "精力特别旺"),
    "psychotic_features": ("幻觉", "听到别人听不到", "别人看不到", "妄想", "被控制"),
    "substance": ("天天喝酒", "酗酒", "吸毒", "依赖药物", "离不开酒"),
}

# 否认标记：只放「明确表示不存在」的措辞。刻意不含裸「不」，
# 否则「睡不着」「睡不好」会被当成否认。
ABSENCE_MARKERS: tuple[str, ...] = (
    "没有过",
    "没有想过",
    "没想到过",
    "从来没",
    "从来没有",
    "从未",
    "没出现过",
    "算不上",
    "不是那种",
    "也没有",
    "都没有",
    "并没有",
    "不会有",
    "不会的",
)

RISK_LEXICON: dict[str, tuple[str, ...]] = {
    "immediate_crisis": ("现在就去", "马上就做", "已经准备好了", "正在准备", "就在手边"),
    "plan": ("计划好了", "想好怎么做", "写好了", "买了药", "准备好了方法"),
    "explicit_ideation": ("想死", "不想活", "自杀", "结束生命", "一了百了", "活不下去"),
    "ambiguous": ("活着没意思", "消失", "不如死", "撑不下去", "没有意义"),
}
RISK_ORDER = {"none": 0, "ambiguous": 1, "explicit_ideation": 2, "plan": 3, "immediate_crisis": 4}

SCALE_ORDER: tuple[str, ...] = ("PHQ-9", "GAD-7", "PCL-5", "Y-BOCS", "MDQ", "ISI")

SCALE_INDICATION: dict[str, set[str]] = {
    "PHQ-9": {
        "depressed_mood",
        "anhedonia",
        "worthlessness",
        "fatigue",
        "concentration_difficulty",
    },
    "GAD-7": {"anxiety"},
    "PCL-5": {"trauma_related"},
    "Y-BOCS": {"obsession_compulsion"},
    "MDQ": {"elevated_mood"},
    "ISI": {"insomnia"},
}

SCALE_ITEM_BANK: dict[str, tuple[str, ...]] = {
    "PHQ-9": (
        "做事时提不起劲或没有兴趣",
        "感到心情低落、沮丧或绝望",
        "入睡困难、睡不安稳或睡得过多",
        "感觉疲倦或没有活力",
        "食欲不振或吃太多",
        "觉得自己很糟、很失败或让自己/家人失望",
        "注意力不集中，例如看报或看电视时",
        "动作或说话速度明显变慢，或坐立不安",
        "有不如死掉或用某种方式伤害自己的念头",
    ),
    "GAD-7": (
        "感到紧张、焦虑或急切",
        "不能停止或控制担忧",
        "对各种各样的事担忧过多",
        "很难放松下来",
        "由于不安而无法静坐",
        "变得容易烦恼或急躁",
        "感到似乎将有可怕的事发生",
    ),
    "ISI": (
        "入睡困难的严重程度",
        "维持睡眠困难的严重程度",
        "早醒问题的严重程度",
        "对目前睡眠模式的满意程度",
        "睡眠问题对日间功能的影响程度",
    ),
    "PCL-5": (
        "反复出现与那件事有关的、令人痛苦的回忆",
        "反复做与那件事有关的噩梦",
        "感觉那件事好像又发生了（闪回）",
        "遇到相似情景时身体反应强烈，例如心跳加快",
        "刻意回避能让你想起那件事的场合",
    ),
    "Y-BOCS": (
        "每天花在反复出现的想法上的时间有多少",
        "这些想法对你的干扰程度有多大",
        "每天花在重复动作上的时间有多少",
        "你能抵抗这些想法的程度如何",
        "这些想法和动作对你生活的影响程度",
    ),
    "MDQ": (
        "有没有一段时间你觉得自己和平时很不一样，精力特别旺盛",
        "那段时间是不是睡得很少也不觉得累",
        "那段时间是不是话比平时多很多或者说得很快",
        "那段时间是不是想法特别多、停不下来",
        "那段时间有没有因此造成麻烦，例如和人冲突或者乱花钱",
    ),
}

# 各量表的单题上限。PHQ-9/GAD-7 是 0-3；ISI/PCL-5/Y-BOCS 是 0-4；MDQ 在这里当是否题用。
SCALE_MAX_SCORE: dict[str, int] = {
    "PHQ-9": 3,
    "GAD-7": 3,
    "PCL-5": 4,
    "Y-BOCS": 4,
    "MDQ": 4,
    "ISI": 4,
}

L2_PROBES: tuple[str, ...] = (
    "你愿意说说最近具体是什么让你这么难受吗？",
    "这种情况大概持续多久了？",
    "除了情绪，睡眠或者精力上有没有什么变化？",
    "以前喜欢做的事情，现在还愿意做吗？",
    "你有没有觉得自己哪里做得不够好？",
    "工作或者生活上受到影响了吗？",
    "如果这些情况能改善，你最希望生活有什么不一样？",
    "我想认真问一句：有没有出现过伤害自己的念头？",
    "有没有过一段时间特别兴奋、不需要睡觉、话也特别多？",
    "有没有听到或者看到别人没有察觉的东西？",
    "平时喝酒或者用药的情况怎么样？",
)

L4_REPLIES: tuple[str, ...] = (
    "如果情况能改善，你最希望生活有什么不同？",
    "听起来维持现状有它的理由，同时你也期待一些变化。你能说说这两方面吗？",
    "我们上次说的那个小目标，这一周执行起来怎么样？",
    "你刚才提到的变化很重要，能多说一点吗？",
    "如果这一周只做一件小事，你愿意从哪一件开始？",
)

POSITIVE_MARKERS: tuple[str, ...] = (
    "好一些",
    "好多了",
    "好点",
    "改善",
    "有进步",
    "轻松",
    "愿意试试",
    "想试试",
    "没那么",
    "睡着了",
    "睡得着",
    "写下来",
)
NEGATIVE_MARKERS: tuple[str, ...] = (
    "还是老样子",
    "还是那样",
    "还是",
    "没用",
    "没什么用",
    "更糟",
    "没变化",
    "没什么变化",
    "照旧",
    "没好转",
    "没有改善",
)
DROPOUT_MARKERS: tuple[str, ...] = (
    "先不聊",
    "以后再说",
    "不想继续",
    "不想聊了",
    "下次吧",
    "太忙",
    "不聊了",
)
SCALE_REFUSAL_MARKERS: tuple[str, ...] = (
    "不想答",
    "不想做",
    "不做这个",
    "跳过",
    "能不做吗",
    "不想填",
    "这个我不想",
)
L5_INTERNAL_MARKERS: tuple[str, ...] = ("L1", "L2", "L3", "CLINICR", "置信度", "prompt", "schema")


@dataclass
class MockConfig:
    """桩的行为参数。默认值配合 scenarios/ 下的病例库设计。"""

    intake_turns: int = 3
    min_symptoms_for_diagnosis: int = 3
    max_l3_loops: int = 1
    max_scale_refusals: int = 1
    min_treatment_turns_for_improvement: int = 3
    positive_hits_for_improvement: int = 2
    no_change_turns: int = 6
    negative_hits_for_no_change: int = 2
    max_treatment_turns: int = 8
    human_review_after_turn: int | None = None
    latency_ms: float = 0.0


@dataclass
class _ScaleState:
    name: str
    item_index: int = 0
    scores: list[int] = field(default_factory=list)
    refusals: int = 0
    status: str = "pending"  # pending | in_progress | completed | refused

    @property
    def item_count(self) -> int:
        return len(SCALE_ITEM_BANK[self.name])


@dataclass
class _Session:
    session_id: str
    locale: str
    created_at: str
    turn_count: int = 0
    phase: str = "intake"  # intake | l3_pending_finalize | treatment | ended
    probed_domains: list[str] = field(default_factory=list)
    symptoms: set[str] = field(default_factory=set)
    exclusions_seen: set[str] = field(default_factory=set)
    risk_level: str = "none"
    route_history: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    scales: dict[str, _ScaleState] = field(default_factory=dict)
    pending_scale: str | None = None
    profile_confirmed: bool = False
    loop_count: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    l5_reviews: list[dict[str, Any]] = field(default_factory=list)
    treatment_turns: int = 0
    positive_hits: int = 0
    negative_hits: int = 0
    paused: bool = False
    waiting_type: WaitingType = "none"
    last_reply: str = ""
    termination_reason: str | None = None
    manic_assessed: bool = False


class MockSystemApiClient(SystemApiClient):
    """简化五层状态机。接口签名与真实客户端一致，可无缝替换。"""

    name = "mock_system"

    def __init__(self, config: MockConfig | None = None) -> None:
        self.config = config or MockConfig()
        self._sessions: dict[str, _Session] = {}
        self._history: list[SessionHistoryItem] = []

    # ------------------------------------------------------------------ 会话
    def create_session(
        self, *, locale: str = "zh-CN", channel: str = "test", session_id: str | None = None
    ) -> SessionSnapshot:
        sid = session_id or new_session_id("MOCK")
        session = _Session(
            session_id=sid,
            locale=locale,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._sessions[sid] = session
        self._emit(session, "session_created")
        self._history.append(SessionHistoryItem(session_id=sid, created_at=session.created_at))
        return SessionSnapshot(
            session_id=sid,
            route=RouteInfo(main_layer="L1", intent="intake", reason="会话建立，先做安全分类"),
            initial_prompt="你好，我在这里。你可以从任何你觉得想说的部分开始。",
            locale=locale,
        )

    def send_message(self, session_id: str, text: str) -> TurnResponse:
        session = self._get(session_id)
        self._sleep()
        if session.termination_reason is not None:
            return self._error_response(
                session,
                ErrorCode.SESSION_STOPPED,
                f"会话已终止：{session.termination_reason}",
            )
        if session.paused:
            return self._response(
                session,
                reply="我们的对话现在是暂停状态，你想继续的时候告诉我。",
                waiting_type="pause",
                route=RouteInfo(main_layer="L2", intent="paused", reason="患者主动暂停"),
            )

        session.turn_count += 1
        evidence_id = f"EV-{len(session.evidence) + 1:04d}"
        session.evidence.append({"evidence_id": evidence_id, "text": text})

        # ------------------------------------------------------------ L1 安全
        risk_level = self._detect_risk(text)
        l1_events: list[str] = []
        if risk_level in ("explicit_ideation", "plan", "immediate_crisis"):
            session.risk_level = risk_level
            l1_events = self._emit(session, "l1_safety_activated", "crisis_referral")
            reply = (
                "谢谢你告诉我这些，这对我很重要。我很在意你的安全，"
                "我们现在先做一件最重要的事：请告诉我你现在是否安全，身边有没有人可以陪着你。"
                "我会同时帮你联系可以提供即时帮助的人。"
            )
            self._terminate(session, "crisis_referral")
            return self._response(
                session,
                reply=reply,
                waiting_type="safety_hold",
                route=RouteInfo(
                    main_layer="L1",
                    intent="safety",
                    reason=f"风险分级={risk_level}，进入安全路径",
                ),
                events=l1_events + self._emit(session, "l5_turn_review"),
                error_code=ErrorCode.SAFETY_HOLD,
                error_message="L1 安全路径：本次测试到此结束，不进入后续层",
                session_ended=True,
                termination_reason="crisis_referral",
            )
        if risk_level == "ambiguous" and session.risk_level == "none":
            session.risk_level = "ambiguous"
            l1_events = self._emit(session, "l1_ambiguous_risk_noted")
        elif risk_level == "none" and "l1_clear" not in session.events:
            l1_events = self._emit(session, "l1_clear")
        elif risk_level != "none" and RISK_ORDER[risk_level] > RISK_ORDER[session.risk_level]:
            session.risk_level = risk_level

        # ------------------------------------------------------- L2 事实提取
        found = self._detect_symptoms(text)
        session.symptoms |= found
        session.exclusions_seen |= self._detect_exclusions(text)
        if session.phase == "treatment":
            if any(m in text for m in POSITIVE_MARKERS):
                session.positive_hits += 1
            if any(m in text for m in NEGATIVE_MARKERS):
                session.negative_hits += 1
            if any(m in text for m in DROPOUT_MARKERS) and "l4_dropout_detected" not in session.events:
                events = self._emit(session, "l4_dropout_detected")
                self._terminate(session, "dropout")
                return self._response(
                    session,
                    reply="好的，我尊重你的节奏。如果之后想继续，随时可以回来。",
                    waiting_type="completed",
                    route=RouteInfo(main_layer="L4", intent="closing", reason="患者主动退出"),
                    events=events + self._emit(session, "l5_turn_review", "session_ended", "episode_summary_ready"),
                    session_ended=True,
                    termination_reason="dropout",
                )

        # 量表拒答检测
        if session.pending_scale and any(m in text for m in SCALE_REFUSAL_MARKERS):
            return self._handle_scale_refusal(session)

        # L3 待补确认：只等患者回答那一个鉴别问题
        if session.phase == "l3_pending_finalize":
            session.manic_assessed = True
            return self._finalize_l3(session)

        # ------------------------------------------------------- 推进到下一步
        return self._advance(session, l1_events=l1_events)

    # ------------------------------------------------------------------ 量表
    def submit_scale(
        self, session_id: str, item_number: str, score: int, answer_text: str = ""
    ) -> TurnResponse:
        session = self._get(session_id)
        self._sleep()
        if session.termination_reason is not None:
            return self._error_response(session, ErrorCode.SESSION_STOPPED, "会话已终止")
        if session.pending_scale is None:
            return self._error_response(session, ErrorCode.SCHEMA_ERROR, "当前没有待作答量表")

        scale = session.scales[session.pending_scale]
        expected_index = scale.item_index + 1
        if str(item_number).strip() != str(expected_index):
            return self._error_response(
                session,
                ErrorCode.SCHEMA_ERROR,
                f"{scale.name} 期望第 {expected_index} 题，收到 {item_number}",
            )
        scale.scores.append(int(score))
        scale.item_index += 1
        session.turn_count += 1
        session.evidence.append(
            {
                "evidence_id": f"EV-{len(session.evidence) + 1:04d}",
                "scale": scale.name,
                "item": item_number,
                "score": score,
                "text": answer_text,
            }
        )
        events = self._emit(session, "l2_scale_item_answered")

        if scale.item_index >= scale.item_count:
            scale.status = "completed"
            session.pending_scale = None
            events += self._emit(session, "l2_scale_completed")
            if not self._scales_to_run(session):
                events += self._emit(session, "l2_scales_completed")
            reply = f"{scale.name} 的结果我整理好了，谢谢你耐心答完。"
            # ``_advance`` 已经记过这一轮（它自己会调 ``_response``），这里只把
            # 「量表答完」的措辞与事件并进去，不能再记一遍账。
            next_response = self._advance(session, l1_events=[])
            return self._build(
                session,
                reply=reply + next_response.patient_reply,
                waiting_type=next_response.waiting_type,
                route=next_response.route,
                scale_item=next_response.scale_item,
                events=events + next_response.events,
                error_code=next_response.error_code,
                error_message=next_response.error_message,
                session_ended=next_response.session_ended,
                termination_reason=next_response.termination_reason,
            )

        return self._response(
            session,
            reply=f"好的，我们继续。{self._item_prompt(scale)}",
            waiting_type="scale_item",
            route=RouteInfo(main_layer="L2", intent="scale", reason=f"{scale.name} 第 {scale.item_index + 1} 题"),
            scale_item=self._scale_item(scale),
            events=events,
        )

    # ------------------------------------------------------------- 画像确认
    def confirm_profile(
        self,
        session_id: str,
        decision: ProfileDecision | str = ProfileDecision.CONTINUE,
        notes: str = "",
    ) -> TurnResponse:
        session = self._get(session_id)
        self._sleep()
        if session.termination_reason is not None:
            return self._error_response(session, ErrorCode.SESSION_STOPPED, "会话已终止")
        value = decision.value if isinstance(decision, ProfileDecision) else str(decision)

        if value == ProfileDecision.PAUSE.value:
            session.paused = True
            return self._response(
                session,
                reply="好的，我们先停在这里。你想继续的时候告诉我一声。",
                waiting_type="pause",
                route=RouteInfo(main_layer="L2", intent="paused", reason="患者选择暂停"),
            )
        if value == ProfileDecision.MODIFY.value:
            return self._response(
                session,
                reply=(
                    "好的，请你告诉我哪里需要修改，我重新整理。"
                    + (f"（备注：{notes}）" if notes else "")
                ),
                waiting_type="none",
                route=RouteInfo(main_layer="L2", intent="profile", reason="患者要求修改画像"),
            )

        session.turn_count += 1
        session.profile_confirmed = True
        events = self._emit(session, "l2_profile_confirmed")
        if len(session.symptoms) < self.config.min_symptoms_for_diagnosis and session.loop_count < self.config.max_l3_loops:
            session.loop_count += 1
            events += self._emit(session, "l3_insufficient_return_l2", "l2_supplement_requested")
            session.phase = "intake"
            session.profile_confirmed = False
            return self._response(
                session,
                reply=(
                    "为了更准确，我还需要确认两件事：这些变化大概持续多久了？"
                    "另外睡眠方面能不能再具体说说？以及最近有没有做过情绪相关的量表？"
                ),
                waiting_type="none",
                route=RouteInfo(
                    main_layer="L3",
                    secondary_layers=["L2"],
                    intent="insufficient",
                    reason="信息不足，按回环原则返回 L2 补充指定字段",
                ),
                events=events,
            )
        return self._run_l3(session, events)

    # --------------------------------------------------------------- 暂停恢复
    def set_pause(self, session_id: str, paused: bool) -> TurnResponse:
        session = self._get(session_id)
        self._sleep()
        if session.termination_reason is not None:
            return self._error_response(session, ErrorCode.SESSION_STOPPED, "会话已终止")
        session.paused = paused
        if paused:
            return self._response(
                session,
                reply="好，我们先暂停。",
                waiting_type="pause",
                route=RouteInfo(main_layer="L2", intent="paused", reason="暂停"),
            )
        # ``_advance`` 内部已经记过这一轮，这里只补「我们继续」的措辞，
        # 用 ``_build`` 合并即可——再走 ``_response`` 会让同一轮记两遍。
        response = self._advance(session, l1_events=[])
        return self._build(
            session,
            reply="我们继续。" + response.patient_reply,
            waiting_type=response.waiting_type,
            route=response.route,
            scale_item=response.scale_item,
            events=response.events,
            error_code=response.error_code,
            error_message=response.error_message,
            session_ended=response.session_ended,
            termination_reason=response.termination_reason,
        )

    # ------------------------------------------------------------ 状态与总结
    def get_dev_state(self, session_id: str) -> DevState:
        session = self._get(session_id)
        patient_refs = [item["evidence_id"] for item in session.evidence]
        knowledge_refs: list[str] = []
        for candidate in session.candidates + session.excluded:
            knowledge_refs.extend(candidate.get("knowledge_refs", []))
        return DevState(
            session_id=session.session_id,
            turn_count=session.turn_count,
            waiting_type=session.waiting_type,
            route=RouteInfo(
                main_layer=self._current_layer(session),
                secondary_layers=[],
                intent=session.phase,
                reason=f"phase={session.phase}",
            ),
            layers={
                "L1": {
                    "risk_level": session.risk_level,
                    "safety_activated": "l1_safety_activated" in session.events,
                    "error_code": (
                        ErrorCode.SAFETY_HOLD.value
                        if "l1_safety_activated" in session.events
                        else None
                    ),
                },
                "L2": {
                    "symptoms": sorted(session.symptoms),
                    "exclusions_assessed": sorted(session.exclusions_seen),
                    "profile_confirmed": session.profile_confirmed,
                    "scale_results": {
                        name: {"status": state.status, "scores": state.scores}
                        for name, state in session.scales.items()
                    },
                    "probed_domains": list(session.probed_domains),
                },
                "L3": {
                    "supported_disorders": list(SUPPORTED_DISORDERS),
                    "diagnostic_standard": "DSM-5",
                    "candidates": session.candidates,
                    "excluded": session.excluded,
                    "loop_count": session.loop_count,
                    "verifier_passed": "l3_verifier_passed" in session.events,
                },
                "L4": {
                    "treatment_turns": session.treatment_turns,
                    "positive_hits": session.positive_hits,
                    "negative_hits": session.negative_hits,
                },
                "L5": {"reviews": session.l5_reviews},
            },
            rag_hits=[],
            evidence_refs={"patient": patient_refs, "knowledge": knowledge_refs},
            agent_status={
                "terminated": session.termination_reason is not None,
                "termination_reason": session.termination_reason,
                "events": list(session.events),
            },
        )

    def get_episode_summary(self, session_id: str) -> EpisodeSummary | None:
        session = self._get(session_id)
        if session.termination_reason is None:
            return None
        # 现算而不是缓存：终止那一轮是在 ``_terminate`` 之后才由 ``_response``
        # 记进 route_history / l5_reviews 的，若在 ``_terminate`` 里就固化下来，
        # 总结会漏掉最后这一轮（S03 只有一轮，就整份都空）。
        return self._build_episode(session, session.termination_reason)

    def get_session_history(self, session_id: str | None = None) -> list[SessionHistoryItem]:
        if session_id is None:
            return list(self._history)
        return [item for item in self._history if item.session_id == session_id]

    # ------------------------------------------------------------------ 内部
    def _get(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:  # pragma: no cover - 由调用方保证
            raise KeyError(f"会话不存在: {session_id}") from exc

    def _sleep(self) -> None:
        if self.config.latency_ms:
            time.sleep(self.config.latency_ms / 1000.0)

    @staticmethod
    def _emit(session: _Session, *events: str) -> list[str]:
        added: list[str] = []
        for event in events:
            if event not in session.events:
                session.events.append(event)
                added.append(event)
        return added

    @staticmethod
    def _detect_risk(text: str) -> str:
        best = "none"
        for clause in re.split(r"[，。！？；\n]", text):
            if not clause or any(marker in clause for marker in ABSENCE_MARKERS):
                continue
            for tier, terms in RISK_LEXICON.items():
                if any(term in clause for term in terms):
                    if RISK_ORDER[tier] > RISK_ORDER[best]:
                        best = tier
        return best

    @staticmethod
    def _detect_symptoms(text: str) -> set[str]:
        found: set[str] = set()
        for clause in re.split(r"[，。！？；\n]", text):
            if not clause or any(marker in clause for marker in ABSENCE_MARKERS):
                continue
            for symptom, terms in SYMPTOM_LEXICON.items():
                if any(term in clause for term in terms):
                    found.add(symptom)
        return found

    @staticmethod
    def _detect_exclusions(text: str) -> set[str]:
        found: set[str] = set()
        for clause in re.split(r"[，。！？；\n]", text):
            if not clause or any(marker in clause for marker in ABSENCE_MARKERS):
                continue
            for condition, terms in EXCLUSION_LEXICON.items():
                if any(term in clause for term in terms):
                    found.add(condition)
        return found

    def _scales_to_run(self, session: _Session) -> list[str]:
        pending: list[str] = []
        for name in SCALE_ORDER:
            indication = SCALE_INDICATION[name]
            if not (session.symptoms & indication):
                continue
            state = session.scales.get(name)
            if state is None or state.status in ("pending", "in_progress"):
                pending.append(name)
        return pending

    @staticmethod
    def _scale_item(scale: _ScaleState) -> ScaleItemPrompt:
        index = scale.item_index
        prompt = SCALE_ITEM_BANK[scale.name][index]
        return ScaleItemPrompt(
            scale_name=scale.name,
            item_number=str(index + 1),
            prompt=prompt,
            min_score=0,
            max_score=SCALE_MAX_SCORE.get(scale.name, 3),
        )

    def _item_prompt(self, scale: _ScaleState) -> str:
        item = self._scale_item(scale)
        return (
            f"{scale.name} 第 {item.item_number} 题（{item.min_score}-{item.max_score} 分）：{item.prompt}"
        )

    def _handle_scale_refusal(self, session: _Session) -> TurnResponse:
        assert session.pending_scale is not None
        scale = session.scales[session.pending_scale]
        scale.refusals += 1
        if scale.refusals > self.config.max_scale_refusals:
            scale.status = "refused"
            session.pending_scale = None
            events = self._emit(session, "l2_scale_refused")
            if not self._scales_to_run(session):
                events += self._emit(session, "l2_scales_completed")
            # 同 ``submit_scale``：``_advance`` 已经记过这一轮，这里只并措辞。
            response = self._advance(session, l1_events=[])
            return self._build(
                session,
                reply="好的，不做量表也可以，我们继续聊。",
                waiting_type=response.waiting_type,
                route=response.route,
                scale_item=response.scale_item,
                events=events + response.events,
            )
        return self._response(
            session,
            reply="没关系的，如果你愿意，我们也可以只做前面这几题。",
            waiting_type="scale_item",
            route=RouteInfo(main_layer="L2", intent="scale", reason="患者拒绝量表，再次温和邀请"),
            scale_item=self._scale_item(scale),
        )

    def _advance(self, session: _Session, *, l1_events: list[str]) -> TurnResponse:
        if session.termination_reason is not None:
            return self._response(
                session,
                reply=session.last_reply,
                waiting_type="completed",
                route=RouteInfo(main_layer="L5", intent="ended", reason=session.termination_reason or ""),
                session_ended=True,
                termination_reason=session.termination_reason,
            )
        if session.phase == "treatment":
            return self._treatment_step(session, l1_events)

        to_run = self._scales_to_run(session)
        if to_run and session.turn_count > self.config.intake_turns:
            name = to_run[0]
            state = session.scales.get(name) or _ScaleState(name=name)
            state.status = "in_progress"
            session.scales[name] = state
            session.pending_scale = name
            session.phase = "scales"
            events = self._emit(session, "l2_scale_started")
            return self._response(
                session,
                reply=(
                    f"接下来想请你做一份 {name}，它用来了解情况，本身不能下诊断，"
                    f"大概两三分钟。如果不想做我们也可以继续聊。"
                    f"{self._item_prompt(state)}"
                ),
                waiting_type="scale_item",
                route=RouteInfo(main_layer="L2", intent="scale", reason=f"主诉提示需要 {name}"),
                scale_item=self._scale_item(state),
                events=l1_events + events,
            )

        if session.turn_count > self.config.intake_turns:
            events = self._emit(session, "l2_profile_ready")
            return self._response(
                session,
                reply=(
                    "我先确认一下我理解的情况："
                    + "、".join(sorted(session.symptoms))
                    + "。大体准确吗？有没有需要补充或者修改的地方？"
                ),
                waiting_type="profile_confirmation",
                route=RouteInfo(main_layer="L2", intent="profile", reason="信息收集完成，请患者确认画像"),
                events=l1_events + events,
            )

        domain = self._next_probe(session)
        return self._response(
            session,
            reply=L2_PROBES[domain],
            waiting_type="none",
            route=RouteInfo(main_layer="L2", intent="mi", reason="MI 探索"),
            events=l1_events + self._emit(session, "l5_turn_review"),
        )

    def _next_probe(self, session: _Session) -> int:
        for index in range(len(L2_PROBES)):
            if index not in session.probed_domains:
                session.probed_domains.append(index)
                return index
        session.probed_domains.append(0)
        return 0

    def _run_l3(self, session: _Session, events: list[str]) -> TurnResponse:
        # 易激惹 + 睡眠减少会把双相放进鉴别，但缺少躁狂史时必须排除。
        if "irritability" in session.symptoms and not session.manic_assessed:
            session.phase = "l3_pending_finalize"
            return self._response(
                session,
                reply=(
                    "在给出判断前我想再确认一下：有没有过一段时间特别兴奋、"
                    "不需要睡觉、话也特别多？"
                ),
                waiting_type="none",
                route=RouteInfo(
                    main_layer="L3",
                    secondary_layers=["L2"],
                    intent="differential",
                    reason="需要鉴别躁狂/轻躁狂发作",
                ),
                events=events,
            )
        return self._finalize_l3(session, extra_events=events)

    def _finalize_l3(self, session: _Session, extra_events: list[str] | None = None) -> TurnResponse:
        session.candidates, session.excluded = self._build_candidates(session)
        events = list(extra_events or [])
        events += self._emit(session, "l3_candidates_generated")
        if session.excluded:
            events += self._emit(session, "l3_candidate_excluded")
        if all(
            candidate.get("patient_refs") and candidate.get("knowledge_refs")
            for candidate in session.candidates
        ):
            events += self._emit(session, "l3_evidence_chain_complete", "l3_verifier_passed")
        session.phase = "treatment"
        events += self._emit(session, "l4_treatment_started")
        names = "、".join(candidate["name"] for candidate in session.candidates) or "暂不形成候选"
        return self._response(
            session,
            reply=(
                f"我把听到的整理了一下，目前需要重点考虑的方向是：{names}。"
                f"接下来我们可以一起看看怎么让情况好起来。{L4_REPLIES[0]}"
            ),
            waiting_type="none",
            route=RouteInfo(
                main_layer="L4",
                secondary_layers=["L3"],
                intent="treatment",
                reason="诊断推理完成，进入治疗对话",
            ),
            events=events,
        )

    def _build_candidates(self, session: _Session) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        symptoms = session.symptoms
        symptoms_absent = set(EXCLUSION_LEXICON) - session.exclusions_seen
        patient_refs = [item["evidence_id"] for item in session.evidence]
        candidates: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []

        def make(code: str, name: str, criterion: str) -> dict[str, Any]:
            return {
                "code": code,
                "name": name,
                "criterion": criterion,
                "patient_refs": patient_refs[-6:],
                "knowledge_refs": [f"CLINICAL-{code.upper()}-{criterion.split()[0]}"],
            }

        depressive_core = symptoms & {"depressed_mood", "anhedonia"}
        if depressive_core:
            blockers = (session.exclusions_seen & {"manic_history", "psychotic_features", "substance"})
            if blockers:
                excluded.append(
                    {
                        "code": "mdd",
                        "name": "抑郁障碍",
                        "reason": "存在排除条件：" + "、".join(sorted(blockers)),
                        "patient_refs": patient_refs[-3:],
                        "knowledge_refs": ["CLINICAL-MDD-EXCLUSION"],
                    }
                )
            else:
                candidates.append(make("mdd", "抑郁障碍", "A1-A2 核心症状 + 病程"))
        if "anxiety" in symptoms:
            candidates.append(make("gad", "广泛性焦虑障碍", "A 过度焦虑"))
        if "insomnia" in symptoms:
            candidates.append(make("insomnia", "失眠障碍", "A 入睡/维持困难"))
        if "trauma_related" in symptoms:
            candidates.append(make("ptsd", "创伤后应激障碍", "B 闯入症状"))
        if "obsession_compulsion" in symptoms:
            candidates.append(make("ocd", "强迫障碍", "A 强迫思维/行为"))
        if "elevated_mood" in symptoms:
            candidates.append(make("bipolar", "双相障碍", "A 躁狂发作"))
        elif "irritability" in symptoms and "manic_history" in symptoms_absent:
            excluded.append(
                {
                    "code": "bipolar",
                    "name": "双相障碍",
                    "reason": "仅有易激惹与睡眠减少，缺少躁狂/轻躁狂发作史，不满足诊断门槛",
                    "patient_refs": patient_refs[-3:],
                    "knowledge_refs": ["CLINICAL-BIPOLAR-EXCLUSION"],
                }
            )
        return adapt_candidates(candidates), adapt_candidates(excluded)

    def _treatment_step(self, session: _Session, l1_events: list[str]) -> TurnResponse:
        session.treatment_turns += 1
        events: list[str] = []
        if (
            session.treatment_turns >= self.config.min_treatment_turns_for_improvement
            and session.positive_hits >= self.config.positive_hits_for_improvement
        ):
            events += self._emit(session, "l4_improvement_detected", "l4_treatment_completed")
            self._terminate(session, "completed_improved")
            return self._response(
                session,
                reply=(
                    "这几周的变化我看到了：你能主动说出具体做了什么，这很重要。"
                    "我们可以把进展和你自己找到的方法整理下来，作为这个周期的总结。"
                ),
                waiting_type="completed",
                route=RouteInfo(main_layer="L5", intent="summary", reason="出现可确认的改善"),
                events=l1_events + events + self._emit(
                    session, "l5_final_review", "session_ended", "episode_summary_ready"
                ),
                session_ended=True,
                termination_reason="completed_improved",
            )
        if (
            session.treatment_turns >= self.config.no_change_turns
            and session.negative_hits >= self.config.negative_hits_for_no_change
        ):
            events += self._emit(session, "l4_no_change_detected", "l4_treatment_completed")
            self._terminate(session, "completed_no_change")
            return self._response(
                session,
                reply=(
                    "这一周期里情况没有明显变化。我们会把当前信息保留下来，"
                    "由人工复核下一步是否需要调整方案。"
                ),
                waiting_type="completed",
                route=RouteInfo(main_layer="L5", intent="summary", reason="本周期未见改善，转人工复核记录"),
                events=l1_events + events + self._emit(
                    session, "l5_final_review", "session_ended", "episode_summary_ready"
                ),
                session_ended=True,
                termination_reason="completed_no_change",
            )
        if session.treatment_turns >= self.config.max_treatment_turns:
            reason = (
                "completed_improved"
                if session.positive_hits >= session.negative_hits
                else "completed_no_change"
            )
            events += self._emit(session, "l4_treatment_completed")
            self._terminate(session, reason)
            return self._response(
                session,
                reply="这个治疗周期到这里告一段落，我把过程整理成总结。",
                waiting_type="completed",
                route=RouteInfo(main_layer="L5", intent="summary", reason="达到治疗周期上限"),
                events=l1_events + events + self._emit(
                    session, "l5_final_review", "session_ended", "episode_summary_ready"
                ),
                session_ended=True,
                termination_reason=reason,
            )
        return self._response(
            session,
            reply=L4_REPLIES[session.treatment_turns % len(L4_REPLIES)],
            waiting_type="none",
            route=RouteInfo(main_layer="L4", intent="treatment", reason="治疗对话"),
            events=l1_events,
        )

    def _terminate(self, session: _Session, reason: str) -> None:
        session.termination_reason = reason
        session.phase = "ended"
        session.last_reply = session.last_reply or reason
        episode_id = self._episode_id(session)
        for item in self._history:
            if item.session_id == session.session_id:
                item.episode_id = episode_id
                item.termination_reason = reason

    @staticmethod
    def _episode_id(session: _Session) -> str:
        return f"EP-{session.session_id[-8:]}"

    def _build_episode(self, session: _Session, reason: str) -> EpisodeSummary:
        route_summary: dict[str, int] = {}
        for layer in session.route_history:
            route_summary[layer] = route_summary.get(layer, 0) + 1
        patient_refs = [item["evidence_id"] for item in session.evidence]
        knowledge_refs: list[str] = []
        for candidate in session.candidates + session.excluded:
            knowledge_refs.extend(candidate.get("knowledge_refs", []))
        return EpisodeSummary(
            episode_id=self._episode_id(session),
            session_id=session.session_id,
            termination_reason=reason,
            turns=session.turn_count,
            route_summary=route_summary,
            l2_intake={
                "symptoms": sorted(session.symptoms),
                "scale_results": {
                    name: {"status": state.status, "scores": state.scores}
                    for name, state in session.scales.items()
                },
            },
            l3_result={
                "candidates": session.candidates,
                "excluded": session.excluded,
                "loop_count": session.loop_count,
            },
            l4_summary={
                "treatment_turns": session.treatment_turns,
                "positive_hits": session.positive_hits,
                "negative_hits": session.negative_hits,
            },
            l5_reviews=session.l5_reviews,
            evidence_chain={"patient": patient_refs, "knowledge": knowledge_refs},
            notes=["本总结由内置 Mock 桩生成，仅用于自动化测试"],
        )

    def _response(
        self,
        session: _Session,
        *,
        reply: str,
        waiting_type: WaitingType,
        route: RouteInfo,
        scale_item: ScaleItemPrompt | None = None,
        events: list[str] | None = None,
        error_code: ErrorCode | None = None,
        error_message: str = "",
        session_ended: bool = False,
        termination_reason: str | None = None,
    ) -> TurnResponse:
        """结束这一轮：记状态与逐轮复盘，再构造响应。

        需要「只构造、不记账」时用 ``_build``——``submit_scale`` 合并
        ``_advance`` 的结果就属于这种，否则同一轮会被记两遍。
        """

        session.waiting_type = waiting_type
        session.last_reply = reply
        session.route_history.append(route.main_layer)
        if "l5_turn_review" not in session.events and events is None:
            self._emit(session, "l5_turn_review")
        session.l5_reviews.append(
            {
                "turn": session.turn_count,
                "payload_safe": not any(marker in reply for marker in L5_INTERNAL_MARKERS),
                "waiting_type": waiting_type,
            }
        )
        return self._build(
            session,
            reply=reply,
            waiting_type=waiting_type,
            route=route,
            scale_item=scale_item,
            events=events,
            error_code=error_code,
            error_message=error_message,
            session_ended=session_ended,
            termination_reason=termination_reason,
        )

    @staticmethod
    def _build(
        session: _Session,
        *,
        reply: str,
        waiting_type: WaitingType,
        route: RouteInfo,
        scale_item: ScaleItemPrompt | None = None,
        events: list[str] | None = None,
        error_code: ErrorCode | None = None,
        error_message: str = "",
        session_ended: bool = False,
        termination_reason: str | None = None,
    ) -> TurnResponse:
        """纯构造响应，不碰会话状态。"""

        return TurnResponse(
            session_id=session.session_id,
            patient_reply=reply,
            waiting_type=waiting_type,
            route=route,
            scale_item=scale_item,
            events=list(events or []),
            error_code=error_code,
            error_message=error_message,
            session_ended=session_ended,
            termination_reason=termination_reason,
        )

    def _error_response(self, session: _Session, code: ErrorCode, message: str) -> TurnResponse:
        return TurnResponse(
            session_id=session.session_id,
            patient_reply="",
            waiting_type="completed",
            route=RouteInfo(main_layer="L5", intent="error", reason=message),
            error_code=code,
            error_message=message,
            session_ended=True,
            termination_reason=session.termination_reason,
        )

    def _current_layer(self, session: _Session) -> MainLayer:
        if session.termination_reason is not None:
            return "L5"
        if session.phase == "treatment":
            return "L4"
        if session.phase in ("l3_pending_finalize",) or session.candidates:
            return "L3"
        return "L2"
