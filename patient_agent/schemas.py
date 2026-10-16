"""PA1：PatientScenario 病例 Schema。

对应任务说明第四节「病例模型」的七组字段：

    基本背景 / 临床事实 / 心理特征 / 风险脚本 / 披露策略 / 行为参数 / 变化轨迹

另外补两组，服务于自动化测试而不是临床：

    expected  期望路径事件、期望经过的层、期望终止结果（PA7 评测用）
    stop      PA6 停止条件

约定：
* 所有模型 ``extra="forbid"``，病例里写错字段名会直接报错，不静默忽略。
* 「患者会说的话」一律写成病例里的 utterance 模板，生成器只允许照抄或改写。
  这样「不添加病例外事实」才有可检查的边界。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from patient_agent.lexicon import extract_number_tokens

DisclosureMode = Literal["proactive", "on_probe", "deep_probe", "withhold"]
RiskLevel = Literal["none", "ambiguous", "explicit_ideation", "plan", "immediate_crisis"]
TrajectoryKind = Literal["improving", "stable", "worsening", "relapse", "dropout"]
TerminationReason = Literal[
    "crisis_referral",
    "human_review",
    "completed_improved",
    "completed_no_change",
    "relapse_referral",
    "dropout",
    "system_error",
]
LayerName = Literal["L1", "L2", "L3", "L4", "L5", "orchestrator"]
Verbosity = Literal["terse", "medium", "verbose"]
Willingness = Literal["accept", "hesitate", "refuse"]
Sentiment = Literal["positive", "negative", "neutral"]
FactKind = Literal["risk", "symptom", "exclusion", "background", "psychology"]


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 11】共享 Schema：PatientScenario 及其子模型
# 接入什么：任务说明要求先冻结的共享 Schema 之一（RetrievedKnowledge / L2KnowledgeCard /
#           ClinicalCriterion / PatientScenario 等）。本工作包只负责 PatientScenario
#           及以下（病例侧的 7+2 组字段）。
# 何时需要改：若其它工作包调整了共享字段，病例 JSON、library.py、evaluator.py 都会跟着
#           受影响，改动要一起同步，否则病例库会在加载时直接报错。
# 注意：所有模型 extra="forbid"，新增字段必须同时在 schema 与所有病例文件里落地。
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioModel(BaseModel):
    """病例相关模型的公共基类：拒绝未知字段。"""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------- 基本背景
class BackgroundFact(ScenarioModel):
    """一条可披露的背景细节（职业、家庭、关键事件）。"""

    fact_id: str
    field: str
    content: str = Field(min_length=1)
    disclosure: DisclosureMode = "on_probe"
    probe_keywords: list[str] = Field(default_factory=list)
    # 只有系统先抛出这些事件之后，该事实才可被披露（例如 L3→L2 回环追问后才说的补充）。
    unlock_after_events: list[str] = Field(default_factory=list)


class BasicBackground(ScenarioModel):
    age_range: str
    occupation: str
    family_relationship: str
    major_events: list[str] = Field(default_factory=list)
    facts: list[BackgroundFact] = Field(default_factory=list)
    # 与背景矛盾的表述，例如在职却说「我已经辞职了」。
    forbidden_claims: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- 临床事实
class SymptomFact(ScenarioModel):
    symptom_id: str
    label: str
    present: bool
    duration: str = "unknown"
    severity: int = Field(default=1, ge=0, le=3)
    utterance: str = ""
    # present=False 时，被追问该怎么否认。
    denial_utterance: str = ""
    disclosure: DisclosureMode = "on_probe"
    probe_keywords: list[str] = Field(default_factory=list)
    functional_impact: list[str] = Field(default_factory=list)
    unlock_after_events: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_utterances(self) -> "SymptomFact":
        if self.present and not self.utterance:
            raise ValueError(f"症状 {self.symptom_id} 标记为存在，必须给出 utterance")
        if not self.present and not self.denial_utterance:
            raise ValueError(f"症状 {self.symptom_id} 标记为不存在，必须给出 denial_utterance")
        return self


class ExclusionFact(ScenarioModel):
    """需要被系统排除的条件：躁狂史、精神病性症状、物质影响等。"""

    condition_id: str
    label: str
    present: bool
    utterance: str = ""
    denial_utterance: str = ""
    probe_keywords: list[str] = Field(default_factory=list)
    unlock_after_events: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_utterances(self) -> "ExclusionFact":
        if self.present and not self.utterance:
            raise ValueError(f"排除条件 {self.condition_id} 标记为存在，必须给出 utterance")
        if not self.present and not self.denial_utterance:
            raise ValueError(
                f"排除条件 {self.condition_id} 标记为不存在，必须给出 denial_utterance"
            )
        return self


class ScalePlan(ScenarioModel):
    """量表的作答计划。answers 按题号顺序给出每题分值。"""

    scale_name: str
    willingness: Willingness = "accept"
    answers: list[int] = Field(default_factory=list)
    refuse_after_items: int | None = Field(default=None, ge=1)
    comment: str = ""

    @model_validator(mode="after")
    def _check_answers(self) -> "ScalePlan":
        if self.willingness != "refuse" and not self.answers:
            raise ValueError(f"量表 {self.scale_name} 未拒绝作答，必须给出 answers")
        if any(score < 0 or score > 4 for score in self.answers):
            raise ValueError(f"量表 {self.scale_name} 的分值必须在 0-4 之间")
        if self.refuse_after_items is not None and self.refuse_after_items > len(self.answers):
            raise ValueError(f"量表 {self.scale_name} 的中断题号超出 answers 长度")
        return self


from .diagnostic_catalog import diagnostic_codes


class ClinicalFacts(ScenarioModel):
    chief_complaint: str
    symptoms: list[SymptomFact] = Field(min_length=1)
    exclusions: list[ExclusionFact] = Field(default_factory=list)
    scale_plans: list[ScalePlan] = Field(default_factory=list)
    # 病程锚点，例如 {"onset": "2个月"}；守卫用它核对患者不会前后说出两个病程。
    timeline_anchors: dict[str, str] = Field(default_factory=dict)
    expected_diagnosis_codes: list[str] = Field(default_factory=list)

    @field_validator("expected_diagnosis_codes", mode="before")
    @classmethod
    def canonical_expected_diagnoses(cls, values):
        return list(dict.fromkeys(code for value in values or [] for code in diagnostic_codes(value)))

    # 系统不许下的诊断（验证「候选被排除」）。
    must_not_diagnose: list[str] = Field(default_factory=list)
    # 允许患者自己提到的诊断名（例如「社区医生说我是抑郁」），默认不允许。
    patient_reported_labels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique(self) -> "ClinicalFacts":
        symptom_ids = [item.symptom_id for item in self.symptoms]
        if len(symptom_ids) != len(set(symptom_ids)):
            raise ValueError("symptoms 里的 symptom_id 必须唯一")
        condition_ids = [item.condition_id for item in self.exclusions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("exclusions 里的 condition_id 必须唯一")
        plan_names = [item.scale_name for item in self.scale_plans]
        if len(plan_names) != len(set(plan_names)):
            raise ValueError("scale_plans 里的 scale_name 必须唯一")
        if not any(item.present for item in self.symptoms):
            raise ValueError("至少要有一种症状 present=True，否则病例没有可模拟的内容")
        return self


# --------------------------------------------------------------------- 心理特征
class EmotionFact(ScenarioModel):
    emotion: str
    intensity: int = Field(default=5, ge=0, le=10)
    utterance: str = ""


class PsychologicalProfile(ScenarioModel):
    emotions: list[EmotionFact] = Field(default_factory=list)
    cognitive_patterns: list[str] = Field(default_factory=list)
    core_beliefs: list[str] = Field(default_factory=list)
    goal: str = ""
    importance: int | None = Field(default=None, ge=0, le=10)
    confidence: int | None = Field(default=None, ge=0, le=10)
    readiness: int | None = Field(default=None, ge=0, le=10)
    ambivalence: str = ""


# --------------------------------------------------------------------- 风险脚本
class RiskScript(ScenarioModel):
    level: RiskLevel = "none"
    utterance: str = ""
    # 无风险病例被问到安全问题时怎么否认。
    denial_utterance: str = ""
    disclosure: DisclosureMode = "proactive"
    probe_keywords: list[str] = Field(default_factory=list)
    unlock_after_events: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "RiskScript":
        if self.level == "none":
            if not self.denial_utterance:
                raise ValueError("risk.level=none 时必须提供 denial_utterance")
        elif not self.utterance:
            raise ValueError(f"risk.level={self.level} 时必须提供 utterance")
        return self


# --------------------------------------------------------------------- 披露策略
class DisclosurePolicy(ScenarioModel):
    opening_templates: list[str] = Field(min_length=1)
    max_new_facts_per_turn: int = Field(default=1, ge=1, le=3)
    require_probe_for_protected: bool = True
    refusal_utterances: list[str] = Field(default_factory=list)
    # 话题名；命中且随机数不足时患者会拒答。
    refusable_topics: list[str] = Field(default_factory=list)
    never_disclose: list[str] = Field(default_factory=list)
    # 不含任何病例事实的过渡话术，用于「这一轮没有新信息」的情况。
    non_fact_replies: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- 行为参数
class BehaviorParams(ScenarioModel):
    cooperation: float = Field(default=0.7, ge=0.0, le=1.0)
    verbosity: Verbosity = "medium"
    emotional_intensity: int = Field(default=5, ge=0, le=10)
    repetition: float = Field(default=0.2, ge=0.0, le=1.0)
    hostility: float = Field(default=0.0, ge=0.0, le=1.0)
    ask_back: float = Field(default=0.1, ge=0.0, le=1.0)


# ----------------------------------------------------- 画像确认 / 暂停恢复计划
class ProfileConfirmationPlan(ScenarioModel):
    """系统请患者确认画像时，这个病例怎么答。

    ``decisions`` 是**按顺序**取用的：第一次被问用第 1 个，之后一直用最后一个。
    这样「先要求修改、再确认」这类组合用一条配置就能表达。
    """

    decisions: list[Literal["continue", "modify", "pause"]] = Field(
        default_factory=lambda: ["continue"], min_length=1
    )
    notes: str = ""

    def decision_for(self, asked_index: int) -> str:
        if asked_index < 0:
            asked_index = 0
        return self.decisions[min(asked_index, len(self.decisions) - 1)]


class PausePlan(ScenarioModel):
    """患者主动暂停/恢复的计划，用于验证暂停-恢复链路。"""

    # 第几轮 send_message 之后发起暂停（1 起算）；None 表示不主动暂停。
    pause_after_turn: int | None = Field(default=None, ge=1)
    resume_after_pause: bool = True
    pause_note: str = ""
    resume_note: str = ""


# --------------------------------------------------------------------- 变化轨迹
class TrajectoryStep(ScenarioModel):
    """轨迹推进点。``after_l4_turn`` 是「治疗层第几轮之后」，
    因为量表逐题作答会把总轮数拉长，用绝对轮次无法稳定描述疗程推进。"""

    after_l4_turn: int = Field(default=1, ge=1)
    kind: TrajectoryKind
    utterance: str = Field(min_length=1)
    sentiment: Sentiment = "neutral"


class Trajectory(ScenarioModel):
    kind: TrajectoryKind = "stable"
    steps: list[TrajectoryStep] = Field(default_factory=list)


# --------------------------------------------------------------------- 期望结果
class ExpectedEvent(ScenarioModel):
    event_id: str
    description: str = ""
    layer: LayerName = "orchestrator"
    required: bool = True


class ExpectedOutcome(ScenarioModel):
    # None 表示这个病例不要求观察到疗程终止（例如只验证 L3→L2 回环，
    # 覆盖到目标事件后由停止条件收尾）。
    termination_reason: TerminationReason | None = None
    events: list[ExpectedEvent] = Field(default_factory=list)
    # 一旦出现就说明路径走错了的事件，例如危机样例不该出现任何 L2~L4 事件。
    forbidden_events: list[str] = Field(default_factory=list)
    # 必须出现在 L3 excluded 列表里的候选（验证「候选被排除」）。
    expected_excluded_conditions: list[str] = Field(default_factory=list)

    @field_validator("expected_excluded_conditions", mode="before")
    @classmethod
    def canonical_exclusions(cls, values):
        return list(dict.fromkeys(code for value in values or [] for code in diagnostic_codes(value)))

    expected_layers: list[LayerName] = Field(default_factory=list)
    min_turns: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_unique(self) -> "ExpectedOutcome":
        ids = [item.event_id for item in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("expected.events 里的 event_id 必须唯一")
        return self


# --------------------------------------------------------------------- 停止条件
class StopPolicy(ScenarioModel):
    max_turns: int = Field(default=30, ge=1, le=200)
    stop_on_crisis: bool = True
    stop_on_human_review: bool = True
    stop_on_violation: bool = True
    # 覆盖这些事件后即可停止（测试级停止，不代表疗程终止）。
    target_events: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- 可披露项
class RevealableFact(ScenarioModel):
    """由病例派生出的「一句话事实」，生成器按它决定这一轮说什么。"""

    fact_id: str
    kind: FactKind
    priority: int
    utterance: str = ""
    denial_utterance: str = ""
    denial_expected: bool = False
    probe_keywords: list[str] = Field(default_factory=list)
    disclosure: DisclosureMode = "on_probe"
    # 空表示立即可用；非空表示必须先观测到其中任一系统事件。
    unlock_after_events: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- 病例
class PatientScenario(ScenarioModel):
    scenario_id: str
    version: str = "1.0.0"
    title: str
    description: str = ""
    locale: str = "zh-CN"
    seed: int = 17
    tags: list[str] = Field(default_factory=list)

    background: BasicBackground
    clinical: ClinicalFacts
    psychology: PsychologicalProfile
    risk: RiskScript
    disclosure: DisclosurePolicy
    behavior: BehaviorParams
    trajectory: Trajectory
    profile_confirmation: ProfileConfirmationPlan = Field(
        default_factory=ProfileConfirmationPlan
    )
    pause_plan: PausePlan = Field(default_factory=PausePlan)

    expected: ExpectedOutcome
    stop: StopPolicy = Field(default_factory=StopPolicy)

    @model_validator(mode="after")
    def _check(self) -> "PatientScenario":
        if self.expected.min_turns > self.stop.max_turns:
            raise ValueError("expected.min_turns 不能大于 stop.max_turns")
        return self

    # ------------------------------------------------------------- 派生视图
    def revealable_facts(self) -> list[RevealableFact]:
        """把七组字段摊平成一串「可说的事实」，供生成器调度。"""

        facts: list[RevealableFact] = []
        if self.risk.level != "none":
            facts.append(
                RevealableFact(
                    fact_id="risk:main",
                    kind="risk",
                    priority=0,
                    utterance=self.risk.utterance,
                    probe_keywords=list(self.risk.probe_keywords),
                    disclosure=self.risk.disclosure,
                    unlock_after_events=list(self.risk.unlock_after_events),
                )
            )
        else:
            facts.append(
                RevealableFact(
                    fact_id="risk:absent",
                    kind="risk",
                    priority=90,
                    denial_utterance=self.risk.denial_utterance,
                    denial_expected=True,
                    probe_keywords=list(self.risk.probe_keywords),
                    disclosure="on_probe",
                    unlock_after_events=list(self.risk.unlock_after_events),
                )
            )

        present_order = 0
        for index, symptom in enumerate(self.clinical.symptoms):
            if symptom.present:
                priority = 5 if present_order == 0 else 10 + index
                present_order += 1
                facts.append(
                    RevealableFact(
                        fact_id=f"symptom:{symptom.symptom_id}",
                        kind="symptom",
                        priority=priority,
                        utterance=symptom.utterance,
                        probe_keywords=list(symptom.probe_keywords),
                        disclosure=symptom.disclosure,
                        unlock_after_events=list(symptom.unlock_after_events),
                    )
                )
            else:
                facts.append(
                    RevealableFact(
                        fact_id=f"symptom_absent:{symptom.symptom_id}",
                        kind="symptom",
                        priority=15 + index,
                        denial_utterance=symptom.denial_utterance,
                        denial_expected=True,
                        probe_keywords=list(symptom.probe_keywords),
                        disclosure=symptom.disclosure,
                        unlock_after_events=list(symptom.unlock_after_events),
                    )
                )

        for index, exclusion in enumerate(self.clinical.exclusions):
            if exclusion.present:
                facts.append(
                    RevealableFact(
                        fact_id=f"exclusion:{exclusion.condition_id}",
                        kind="exclusion",
                        priority=30 + index,
                        utterance=exclusion.utterance,
                        probe_keywords=list(exclusion.probe_keywords),
                        disclosure="on_probe",
                        unlock_after_events=list(exclusion.unlock_after_events),
                    )
                )
            else:
                facts.append(
                    RevealableFact(
                        fact_id=f"exclusion_absent:{exclusion.condition_id}",
                        kind="exclusion",
                        priority=20 + index,
                        denial_utterance=exclusion.denial_utterance,
                        denial_expected=True,
                        probe_keywords=list(exclusion.probe_keywords),
                        disclosure="on_probe",
                        unlock_after_events=list(exclusion.unlock_after_events),
                    )
                )

        for index, fact in enumerate(self.background.facts):
            facts.append(
                RevealableFact(
                    fact_id=f"background:{fact.fact_id}",
                    kind="background",
                    priority=40 + index,
                    utterance=fact.content,
                    probe_keywords=list(fact.probe_keywords),
                    disclosure=fact.disclosure,
                    unlock_after_events=list(fact.unlock_after_events),
                )
            )

        psychology_bits = [
            text
            for text in (self.psychology.ambivalence, self.psychology.goal)
            if text
        ]
        if psychology_bits:
            facts.append(
                RevealableFact(
                    fact_id="psychology:motivation",
                    kind="psychology",
                    priority=60,
                    utterance=" ".join(psychology_bits),
                    probe_keywords=["目标", "改变", "希望", "动机", "重要", "打算", "信心"],
                    disclosure="on_probe",
                )
            )
        for index, belief in enumerate(self.psychology.core_beliefs):
            facts.append(
                RevealableFact(
                    fact_id=f"psychology:belief:{index}",
                    kind="psychology",
                    priority=65 + index,
                    utterance=belief,
                    probe_keywords=["觉得自己", "怎么看自己", "价值", "评价"],
                    disclosure="deep_probe",
                )
            )
        return facts

    def declared_numbers(self) -> set[str]:
        """病例里出现过的数字集合。守卫用它近似检查「有没有编造新事实」。"""

        tokens: set[str] = set()
        for text in self.text_corpus():
            tokens |= extract_number_tokens(text)
        for plan in self.clinical.scale_plans:
            tokens |= {str(score) for score in plan.answers}
        for emotion in self.psychology.emotions:
            tokens.add(str(emotion.intensity))
        for value in (
            self.psychology.importance,
            self.psychology.confidence,
            self.psychology.readiness,
        ):
            if value is not None:
                tokens.add(str(value))
        return tokens

    def text_corpus(self) -> list[str]:
        """病例里所有「患者可能说出口」的文本，用于派生词表。"""

        corpus: list[str] = [
            self.background.age_range,
            self.background.occupation,
            self.background.family_relationship,
            *self.background.major_events,
            *self.background.forbidden_claims,
            self.clinical.chief_complaint,
            *self.clinical.timeline_anchors.values(),
        ]
        for fact in self.background.facts:
            corpus.extend([fact.field, fact.content, *fact.probe_keywords])
        for symptom in self.clinical.symptoms:
            corpus.extend(
                [
                    symptom.label,
                    symptom.duration,
                    symptom.utterance,
                    symptom.denial_utterance,
                    *symptom.probe_keywords,
                    *symptom.functional_impact,
                ]
            )
        for exclusion in self.clinical.exclusions:
            corpus.extend(
                [
                    exclusion.label,
                    exclusion.utterance,
                    exclusion.denial_utterance,
                    *exclusion.probe_keywords,
                ]
            )
        for plan in self.clinical.scale_plans:
            corpus.extend([plan.scale_name, plan.comment])
        corpus.extend(
            [
                self.psychology.goal,
                self.psychology.ambivalence,
                *self.psychology.cognitive_patterns,
                *self.psychology.core_beliefs,
            ]
        )
        for emotion in self.psychology.emotions:
            corpus.extend([emotion.emotion, emotion.utterance])
        corpus.extend(
            [
                self.risk.utterance,
                self.risk.denial_utterance,
                *self.risk.probe_keywords,
            ]
        )
        corpus.extend(self.disclosure.opening_templates)
        corpus.extend(self.disclosure.non_fact_replies)
        corpus.extend(self.disclosure.refusal_utterances)
        for step in self.trajectory.steps:
            corpus.append(step.utterance)
        return [text for text in corpus if text]

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "title": self.title,
            "risk_level": self.risk.level,
            "symptoms": [
                item.symptom_id for item in self.clinical.symptoms if item.present
            ],
            "excluded_conditions": [
                item.condition_id
                for item in self.clinical.exclusions
                if not item.present
            ],
            "scales": [item.scale_name for item in self.clinical.scale_plans],
            "trajectory": self.trajectory.kind,
            "expected_termination": self.expected.termination_reason,
        }
