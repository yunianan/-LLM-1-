"""PA4 一致性守卫：患者台词说完之后，先自己检查一遍再发给系统。

为什么需要它：真实模型改写措辞时可能「加戏」——编一个病例里没有的病程、
冒出个诊断名、把风险说得比脚本更重、甚至漏出内部术语。这类输出一旦进入系统，
会污染「不迎合」「不越权诊断」的验证结论，让人分不清是系统的问题还是患者的问题。

硬规则核查事实编号、数量、权限和风险。真实模型模式还独立复核完整回答；
离线模式只允许与病例参考台词一致的文本。任何阻断性违规都不得发送给系统。
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.client.base import ScaleItemPrompt
from patient_agent.generator import PatientAction
from patient_agent.lexicon import (
    DIAGNOSIS_LABELS,
    INTERNAL_JARGON,
    RISK_TIER_ORDER,
    SEVERITY_HIGH_TERMS,
    RISK_TERMS,
    NUMBER_UNIT_RE,
    asserted_risk_tier,
    contains_any,
    extract_number_tokens,
    is_negated_occurrence,
    normalize_quantities,
)
from patient_agent.schemas import PatientScenario
from patient_agent.semantic_guard import SemanticFactVerifier, AnswerReview, ClaimReview
from patient_agent.evidence import FactEvidence, evidence_bundle


class GuardCode(str, Enum):
    DIAGNOSIS_OVERREACH = "DIAGNOSIS_OVERREACH"
    INTERNAL_JARGON_LEAK = "INTERNAL_JARGON_LEAK"
    RISK_ESCALATION = "RISK_ESCALATION"
    FABRICATED_NUMBER = "FABRICATED_NUMBER"
    FACT_MISMATCH = "FACT_MISMATCH"
    FORBIDDEN_CLAIM = "FORBIDDEN_CLAIM"
    NEVER_DISCLOSE = "NEVER_DISCLOSE"
    TIMELINE_CONFLICT = "TIMELINE_CONFLICT"
    SCALE_MISMATCH = "SCALE_MISMATCH"
    SEVERITY_INFLATION = "SEVERITY_INFLATION"
    EMPTY_UTTERANCE = "EMPTY_UTTERANCE"
    UNKNOWN_FACT = "UNKNOWN_FACT"
    DISCLOSURE_VIOLATION = "DISCLOSURE_VIOLATION"
    GUARD_REVIEW_UNAVAILABLE = "GUARD_REVIEW_UNAVAILABLE"
    GENERATION_INVALID = "GENERATION_INVALID"
    EVIDENCE_BINDING_INVALID = "EVIDENCE_BINDING_INVALID"


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: GuardCode
    message: str
    evidence: list[str] = Field(default_factory=list)
    # 阻断性违规不得提交；是否重说由 retryable 和运行器上限共同决定。
    blocking: bool = True
    retryable: bool = True


class GuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    authorized_evidence: list[FactEvidence] = Field(default_factory=list)
    semantic_review: AnswerReview | None = None
    review_mode: str = "not_run"
    semantic_raw_output: str = ""
    negative_evidence: list[FactEvidence] = Field(default_factory=list)
    semantic_request: dict[str, str] = Field(default_factory=dict)
    semantic_prompt_version: str = "patient-review-v1"

    @property
    def blocking(self) -> list[Violation]:
        return [item for item in self.violations if item.blocking]

    def codes(self) -> list[str]:
        return [item.code.value for item in self.violations]


# 严重度分级用词：患者把「偶尔」说成「每天」会让风险看起来更高，属于轻度夸张。
_TIME_UNITS = {"月", "星期", "礼拜", "周", "天", "年"}


class ConsistencyGuard:
    """对照病例脚本检查生成结果。"""

    def __init__(
        self,
        scenario: PatientScenario,
        *,
        allow_severity_upgrade: bool = False,
        verifier: SemanticFactVerifier | None = None,
    ) -> None:
        self.scenario = scenario
        self.allow_severity_upgrade = allow_severity_upgrade
        self.verifier = verifier
        self._facts = {item.fact_id: item for item in scenario.revealable_facts()}
        self._allowed_labels = set(scenario.clinical.patient_reported_labels)
        # 患者被允许说的诊断名：白名单 + 明确禁止自行下的诊断（用于只报「不许下」的情况）。
        self._forbidden_labels = set(scenario.clinical.must_not_diagnose)

    # ------------------------------------------------------------------ 入口
    def check(
        self,
        action: PatientAction,
        *,
        probe_text: str = "",
        scale_item: ScaleItemPrompt | None = None,
        events: list[str] | None = None,
    ) -> GuardResult:
        if action.kind != "say":
            return self._check_non_say(action, scale_item)

        text = action.text.strip()
        violations: list[Violation] = []
        if action.generation_error:
            return GuardResult(ok=False, violations=[Violation(code=GuardCode.GENERATION_INVALID,
                message="结构化生成失败：" + action.generation_error)],
                authorized_evidence=evidence_bundle(self.scenario, action.fact_id))
        if not text:
            violations.append(
                Violation(
                    code=GuardCode.EMPTY_UTTERANCE,
                    message="患者这一轮没有说话内容",
                )
            )
            return GuardResult(ok=False, violations=violations)

        violations += self._check_jargon(text)
        violations += self._check_diagnosis(text)
        violations += self._check_never_disclose(text)
        violations += self._check_forbidden_claims(text)
        references, source_errors = self._references(action, events=events)
        bundle = evidence_bundle(self.scenario, action.fact_id) if not source_errors else []
        violations += source_errors
        if action.authorized_evidence and action.authorized_evidence != bundle:
            violations.append(Violation(code=GuardCode.EVIDENCE_BINDING_INVALID, message="生成依据快照与正式病例不一致"))
        if action.segments:
            if "".join(s.text for s in action.segments) != action.text or len({s.segment_id for s in action.segments}) != len(action.segments):
                violations.append(Violation(code=GuardCode.EVIDENCE_BINDING_INVALID, message="片段未完整组成候选文本或编号重复"))
            for segment in action.segments:
                if ((segment.kind == "factual" and not segment.fact_ids) or
                    (segment.kind == "non_factual" and segment.fact_ids) or
                    any(ref not in references or ref.startswith("neutral:") for ref in segment.fact_ids)):
                    violations.append(Violation(code=GuardCode.EVIDENCE_BINDING_INVALID, message="片段引用无效或缺少依据", evidence=[segment.segment_id]))
        elif action.generation_mode == "structured":
            violations.append(Violation(code=GuardCode.EVIDENCE_BINDING_INVALID, message="结构化生成缺少绑定片段"))
        violations += self._check_numbers(text, references)
        violations += self._check_risk(text, action.is_denial)
        violations += self._check_severity(text, action)
        violations += self._check_timeline(text, references)
        review = None
        mode = "not_run"
        if not any(item.blocking for item in violations):
            errors, review = self._check_fact_consistency(text, references, probe_text, action)
            violations += errors
            mode = "semantic" if self.verifier else "exact_reference"
        return GuardResult(ok=not any(item.blocking for item in violations), violations=violations,
            authorized_evidence=bundle, semantic_review=review, review_mode=mode,
            semantic_raw_output=self.verifier.last_raw_output if self.verifier and mode == "semantic" else "",
            semantic_request=self.verifier.last_request if self.verifier and mode == "semantic" else {},
            negative_evidence=[e for f in self._facts.values() if f.denial_expected
                for e in evidence_bundle(self.scenario, f.fact_id)] if mode == "semantic" else [])

    # -------------------------------------------------------------- 各类检查
    def _check_non_say(
        self, action: PatientAction, scale_item: ScaleItemPrompt | None
    ) -> GuardResult:
        if action.kind == "submit_scale":
            if scale_item is None:
                return GuardResult(ok=False, violations=[Violation(
                    code=GuardCode.SCALE_MISMATCH, message="缺少当前量表题目，不能验证或提交分值", retryable=False,
                )])
            if action.item_number != scale_item.item_number:
                return GuardResult(ok=False, violations=[Violation(
                    code=GuardCode.SCALE_MISMATCH, message="回答题号与当前题目不一致", retryable=False,
                )])
            if not (scale_item.min_score <= action.score <= scale_item.max_score):
                return GuardResult(
                    ok=False,
                    violations=[
                        Violation(
                            code=GuardCode.SCALE_MISMATCH,
                            message=(
                                f"量表 {scale_item.scale_name} 第 {scale_item.item_number} 题"
                                f"给了 {action.score} 分，超出 "
                                f"{scale_item.min_score}-{scale_item.max_score}"
                            ),
                        )
                    ],
                )
            plan = next((p for p in self.scenario.clinical.scale_plans if p.scale_name == scale_item.scale_name), None)
            index = int(action.item_number) - 1 if action.item_number.isdigit() else -1
            if plan is None or index < 0 or index >= len(plan.answers) or plan.willingness == "refuse" or (plan.refuse_after_items is not None and index >= plan.refuse_after_items):
                return GuardResult(ok=False, violations=[Violation(
                    code=GuardCode.SCALE_MISMATCH, message="病例未授权当前量表题目作答", retryable=False,
                )])
            if action.score != plan.answers[index]:
                return GuardResult(ok=False, violations=[Violation(
                    code=GuardCode.SCALE_MISMATCH, message="量表分值与病例预设答案不一致", retryable=False,
                )])
            evidence = FactEvidence(fact_id=f"scale:{plan.scale_name}:item:{action.item_number}",
                scenario_id=self.scenario.scenario_id, scenario_version=self.scenario.version,
                value=str(plan.answers[index]), status="present",
                source_values={f"clinical.scale_plans[{self.scenario.clinical.scale_plans.index(plan)}].answers[{index}]": plan.answers[index]})
            return GuardResult(ok=True, authorized_evidence=[evidence], review_mode="scale_rule")
        return GuardResult(ok=True)

    def _check_jargon(self, text: str) -> list[Violation]:
        hits = [term for term in INTERNAL_JARGON if term in text]
        if not hits:
            return []
        return [
            Violation(
                code=GuardCode.INTERNAL_JARGON_LEAK,
                message="患者台词里出现了系统内部术语，说明生成器发生了结构泄露",
                evidence=hits,
            )
        ]

    def _check_diagnosis(self, text: str) -> list[Violation]:
        violations: list[Violation] = []
        for label in DIAGNOSIS_LABELS:
            if label not in text:
                continue
            if label in self._allowed_labels:
                continue
            # 「医生说我抑郁」这类转述放行；自己下结论不放行。
            if self._is_attributed(text, label):
                continue
            violations.append(
                Violation(
                    code=GuardCode.DIAGNOSIS_OVERREACH,
                    message=f"患者自行给出了诊断名「{label}」，超出病例授权",
                    evidence=[label],
                )
            )
        if not violations:
            hits = [item for item in self._forbidden_labels if item in text]
            if hits and not all(self._is_attributed(text, item) for item in hits):
                violations.append(
                    Violation(
                        code=GuardCode.DIAGNOSIS_OVERREACH,
                        message="患者说出了病例明确禁止自行给出的诊断",
                        evidence=hits,
                    )
                )
        return violations

    @staticmethod
    def _is_attributed(text: str, label: str) -> bool:
        """判断诊断名是不是「转述他人」而非自己下结论。"""

        attribution = ("医生说", "大夫说", "医院说", "社区医生", "别人说", "查出来", "告诉我是")
        window = 12
        start = 0
        while True:
            index = text.find(label, start)
            if index == -1:
                return False
            prefix = text[max(0, index - window) : index]
            if any(marker in prefix for marker in attribution):
                return True
            start = index + len(label)

    def _check_never_disclose(self, text: str) -> list[Violation]:
        risk_words = {term for words in RISK_TERMS.values() for term in words}
        terms = [item for item in self.scenario.disclosure.never_disclose
                 if any(item not in risk_words or not is_negated_occurrence(text, match.start())
                        for match in re.finditer(re.escape(item), text))]
        if not terms:
            return []
        return [
            Violation(
                code=GuardCode.NEVER_DISCLOSE,
                message="患者提到了病例声明为「绝不披露」的内容",
                evidence=terms,
            )
        ]

    def _check_forbidden_claims(self, text: str) -> list[Violation]:
        claims = [item for item in self.scenario.background.forbidden_claims if item in text]
        if not claims:
            return []
        return [
            Violation(
                code=GuardCode.FORBIDDEN_CLAIM,
                message="患者说出了与背景设定矛盾的表述",
                evidence=claims,
            )
        ]

    def _check_numbers(self, text: str, references: dict[str, str]) -> list[Violation]:
        tokens = extract_number_tokens(text)
        allowed = set().union(*(extract_number_tokens(value) for value in references.values()))
        unknown = sorted(tokens - allowed)
        if not unknown:
            return []
        return [
            Violation(
                code=GuardCode.FABRICATED_NUMBER,
                message="回答中的数字或数量没有本轮病例依据（不能借用量表分数等其他数字）",
                evidence=unknown,
                retryable=True,
            )
        ]

    def _check_risk(self, text: str, is_denial: bool) -> list[Violation]:
        declared = RISK_TIER_ORDER[self.scenario.risk.level]
        observed = RISK_TIER_ORDER[asserted_risk_tier(text)]
        if observed > declared:
            return [
                Violation(
                    code=GuardCode.RISK_ESCALATION,
                    message=(
                        f"患者把风险说得比病例脚本更重（脚本={self.scenario.risk.level}，"
                        f"台词={asserted_risk_tier(text)}）"
                    ),
                )
            ]
        return []

    def severity_upgraded(self, text: str) -> bool:
        """患者是否把低频症状说成了高频（用于非阻断的严重度记录）。"""

        if self.allow_severity_upgrade:
            return False
        return contains_any(text, SEVERITY_HIGH_TERMS)

    def _check_severity(self, text: str, action: PatientAction) -> list[Violation]:
        """阻断没有病例原话支持的轻症严重度升级。"""

        if not action.fact_id.startswith("symptom:") or action.is_denial:
            return []
        if self.allow_severity_upgrade:
            return []
        symptom_id = action.fact_id.split(":", 1)[1]
        symptom = next(
            (item for item in self.scenario.clinical.symptoms if item.symptom_id == symptom_id),
            None,
        )
        if symptom is None or not symptom.present or symptom.severity > 1:
            return []
        hits = [term for term in SEVERITY_HIGH_TERMS if term in text and term not in symptom.utterance]
        if not hits:
            return []
        return [
            Violation(
                code=GuardCode.SEVERITY_INFLATION,
                message=(
                    f"症状 {symptom_id} 在病例里是轻度（severity={symptom.severity}），"
                    f"台词用了重度用词"
                ),
                evidence=hits,
            )
        ]

    def _references(self, action: PatientAction, *, events: list[str] | None) -> tuple[dict[str, str], list[Violation]]:
        """仅从正式病例构建依据，不能信任模型生成的 prompt 或参考台词。"""

        if not action.fact_id:
            return {e.fact_id: e.value for e in evidence_bundle(self.scenario, action.fact_id)}, []
        if action.fact_id == "opening":
            return {e.fact_id: e.value for e in evidence_bundle(self.scenario, action.fact_id)}, []
        if action.fact_id.startswith("trajectory:"):
            value = action.fact_id.split(":", 1)[1]
            if value.isdigit() and int(value) < len(self.scenario.trajectory.steps):
                return {e.fact_id: e.value for e in evidence_bundle(self.scenario, action.fact_id)}, []
        fact = self._facts.get(action.fact_id)
        if fact is None:
            return {}, [Violation(code=GuardCode.UNKNOWN_FACT, message="回答引用了不存在的病例事实编号", evidence=[action.fact_id])]
        if fact.disclosure == "withhold" or (events is not None and fact.unlock_after_events and not set(events).intersection(fact.unlock_after_events)):
            return {}, [Violation(code=GuardCode.DISCLOSURE_VIOLATION, message="本轮事实尚未获准披露", evidence=[action.fact_id])]
        if action.is_denial != fact.denial_expected:
            return {}, [Violation(code=GuardCode.FACT_MISMATCH, message="回答的否认标记与正式病例事实不一致", evidence=[action.fact_id])]
        return {e.fact_id: e.value for e in evidence_bundle(self.scenario, action.fact_id)}, []

    def _check_fact_consistency(self, text: str, references: dict[str, str], question: str, action: PatientAction):
        if self.verifier is None:
            normalize = lambda value: re.sub(r"[\s，。！？；：、,.!?;:‘’“”\"'（）()]", "", normalize_quantities(value))
            match = next((ref for ref, value in references.items() if normalize(text) == normalize(value)), None)
            if match:
                if action.segments:
                    claims = []
                    for segment in action.segments:
                        matching = next((r for r, v in references.items() if normalize(v) == normalize(segment.text)), None)
                        if not matching or (segment.kind == "factual" and segment.fact_ids != [matching]) or (segment.kind == "non_factual" and not matching.startswith("neutral:")):
                            return [Violation(code=GuardCode.EVIDENCE_BINDING_INVALID, message="离线模式无法核对片段绑定")], None
                        claims.append(ClaimReview(segment_id=segment.segment_id, quote=segment.text,
                            status="neutral" if matching.startswith("neutral:") else "supported",
                            reference_ids=[] if matching.startswith("neutral:") else [matching], reason="与正式参考台词一致"))
                else:
                    claims = [ClaimReview(quote=text, status="neutral" if match.startswith("neutral:") else "supported",
                        reference_ids=[] if match.startswith("neutral:") else [match], reason="与正式参考台词一致")]
                return [], AnswerReview(claims=claims)
            return [Violation(code=GuardCode.FACT_MISMATCH,
                              message="回答不能由本轮参考台词完整核对，离线守卫拒绝放行",
                              evidence=[text])], None
        try:
            review = self.verifier.verify(text=text, references=references,
                denials=[fact.denial_utterance for fact in self._facts.values() if fact.denial_expected], question=question,
                segments=action.segments)
        except Exception as exc:
            # 必须带上原因：只记异常类型时，报告里看不出是「复核员漏字段」还是「复核员真的判了不支持」，
            # 会把「复核环节没跑完」误读成「患者编造」。
            detail = " ".join(str(exc).split())[:160]
            return [Violation(code=GuardCode.GUARD_REVIEW_UNAVAILABLE,
                              message=f"事实复核不可用或结果不合法：{type(exc).__name__}"
                                      + (f"：{detail}" if detail else "") + "；未放行回答",
                              retryable=False)], None
        return [Violation(code=GuardCode.FACT_MISMATCH,
                          message=f"{claim.status}：{claim.reason}", evidence=[claim.quote])
                for claim in review.claims if claim.status in ("unsupported", "contradicted")], review

    def _check_timeline(self, text: str, references: dict[str, str]) -> list[Violation]:
        def durations(value: str) -> set[str]:
            return {match.group(1) + {"星期": "周", "礼拜": "周"}.get(match.group(2), match.group(2))
                    for match in NUMBER_UNIT_RE.finditer(normalize_quantities(value)) if match.group(2) in _TIME_UNITS}
        allowed = set().union(*(durations(value) for value in references.values()))
        unknown = durations(text) - allowed
        if not unknown:
            return []
        return [Violation(code=GuardCode.TIMELINE_CONFLICT,
                          message="回答含没有本轮依据的时间长度", evidence=sorted(unknown))]
