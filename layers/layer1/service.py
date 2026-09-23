from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from .asq import ASQ_QUESTIONS, handle_asq_answer, start_asq
from .bssa import (
    SAFETY_PLAN_FIELDS,
    apply_extraction,
    determine_disposition,
    emergency_reason,
    interview_bssa,
    missing_fields,
)
from .guard import L1ExitGuard
from .models import L1SessionState, SafetyDisposition, SafetySignalAnalysis
from .policy_executor import execute_safety_policy
from .risk_classifier import L1ModelFailure
from .support import generate_support


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class L1SafetyService:
    """成人门诊L1安全服务；ASQ/BSSA状态只保存在L1。"""

    def __init__(self) -> None:
        self._sessions: dict[str, L1SessionState] = {}
        self.guard = L1ExitGuard()

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def snapshot(self, session_id: str) -> L1SessionState:
        return self._sessions[session_id].model_copy(deep=True)

    def _state(self, session_id: str) -> L1SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = L1SessionState(session_id=session_id)
        return self._sessions[session_id]

    def start_baseline_asq(
        self,
        *,
        session_id: str,
        message_id: str | None = None,
    ) -> dict[str, Any] | None:
        state = self._state(session_id)
        if state.baseline_asq_status in {"negative", "refused", "positive"}:
            return None
        if state.phase != "monitoring":
            return None
        state.baseline_asq_status = "due"
        state.phase = "asq"
        reply = start_asq(state.asq, trigger="baseline")
        return self._passport(
            state,
            message_id=message_id or f"baseline-asq-{uuid.uuid4().hex}",
            decision_path="safety_assessment",
            safety_flag="warn",
            risk_level=2,
            summary="L2参与阶段完成，L1启动常规ASQ筛查。",
            instruction=reply,
            waiting_for="asq_answer",
        )

    def handle_message(
        self,
        text: str,
        *,
        session_id: str,
        message_id: str,
        assessment_context: dict[str, Any] | None = None,
        classifier: Any,
    ) -> dict[str, Any]:
        state = self._state(session_id)
        if state.phase == "asq":
            return self._handle_asq(state, text, message_id)
        if state.phase == "bssa":
            return self._handle_bssa(state, text, message_id)
        if state.phase == "safety_hold":
            return self._held_passport(state, message_id)

        try:
            result = classifier(text, assessment_context or {})
        except L1ModelFailure as exc:
            return self._error_passport(state, message_id, exc)
        analysis = self._analysis(result)
        state.last_signal = analysis

        if analysis.suicide_signal_detected:
            state.phase = "asq"
            reply = start_asq(
                state.asq,
                trigger="signal",
                supportive_opening=analysis.asq_transition_support,
            )
            return self._passport(
                state,
                message_id=message_id,
                decision_path="safety_assessment",
                safety_flag="warn",
                risk_level=2,
                summary="安全语义模型发现自杀风险信号或无法排除风险，启动ASQ。",
                instruction=reply,
                waiting_for="asq_answer",
                evidence=self._signal_evidence(analysis),
            )
        if analysis.jailbreak_detected:
            return self._passport(
                state,
                message_id=message_id,
                decision_path="security_reject",
                safety_flag="pass",
                risk_level=1,
                summary="检测到越狱或提示词攻击请求。",
                instruction="我不能执行绕过安全限制、泄露系统指令或覆盖安全角色的请求。请继续描述您真实需要的帮助。",
                evidence=[
                    {"source": "L1_safety_semantic_model_API", "finding": quote, "basis": "越狱风险证据"}
                    for quote in analysis.jailbreak_evidence
                ],
            )
        return self._passport(
            state,
            message_id=message_id,
            decision_path="normal",
            safety_flag="warn" if state.active_controls else "pass",
            risk_level=2 if state.active_controls else 1,
            summary="未发现需要拦截的自杀风险或越狱风险。",
            instruction="允许进入L2。",
            evidence=self._signal_evidence(analysis),
        )

    @staticmethod
    def _analysis(result: dict[str, Any]) -> SafetySignalAnalysis:
        raw = result.get("analysis")
        if isinstance(raw, dict):
            return SafetySignalAnalysis.model_validate(raw)
        # 仅用于现有受控测试模型的最小兼容；生产API始终返回新Schema。
        risk_types = set(result.get("risk_types") or [])
        score = int(result.get("llm_score", 1))
        detected = bool(risk_types & {"suicidal_ideation", "self_harm"}) or score >= 2
        return SafetySignalAnalysis(
            suicide_signal_detected=detected,
            jailbreak_detected=False,
            safe_to_enter_l2=not detected,
            confidence=float(result.get("confidence", 0.0)),
            evidence_quotes=[],
            jailbreak_evidence=[],
            asq_transition_support="谢谢您愿意告诉我这些。" if detected else "",
            summary="受控模型兼容结果",
        )

    def _handle_asq(
        self, state: L1SessionState, text: str, message_id: str
    ) -> dict[str, Any]:
        turn = handle_asq_answer(state.asq, text)
        if turn.status == "in_progress":
            return self._passport(
                state,
                message_id=message_id,
                decision_path="safety_assessment",
                safety_flag="warn",
                risk_level=2,
                summary=f"ASQ正在进行，等待第{state.asq.current_question}题。",
                instruction=turn.reply,
                waiting_for="asq_answer",
            )
        if turn.status == "negative":
            state.phase = "monitoring"
            state.baseline_asq_status = "negative"
            state.disposition = "no_further_safety_intervention"
            return self._passport(
                state,
                message_id=message_id,
                decision_path="safety_assessment",
                safety_flag="pass",
                risk_level=1,
                summary="ASQ筛查阴性，本轮ASQ答案由L1消费。",
                instruction=turn.reply,
                waiting_for="free_text",
                resume_l2=True,
            )
        if turn.status == "refused":
            # NIMH ASQ 将拒答视为阳性筛查，不能直接交回普通疗程。
            state.baseline_asq_status = "positive"
            return self._start_bssa(
                state,
                message_id=message_id,
                summary="ASQ出现拒答，按阳性筛查进入成人门诊BSSA。",
            )
        state.baseline_asq_status = "positive"
        if turn.status == "acute_positive":
            state.disposition = "emergency_evaluation"
            state.phase = "safety_hold"
            return self._safety_transition(
                state,
                message_id=message_id,
                disposition="emergency_evaluation",
                patient_quote=text,
                basis="ASQ第5题为是，属于急性阳性筛查。",
            )

        return self._start_bssa(
            state,
            message_id=message_id,
            summary="ASQ非急性阳性，进入成人门诊BSSA。",
        )

    def _start_bssa(
        self,
        state: L1SessionState,
        *,
        message_id: str,
        summary: str,
    ) -> dict[str, Any]:
        state.phase = "bssa"
        state.bssa.status = "in_progress"
        try:
            first = interview_bssa("", state.bssa, target="current_ideation")
        except L1ModelFailure as exc:
            return self._error_passport(state, message_id, exc)
        return self._passport(
            state,
            message_id=message_id,
            decision_path="safety_assessment",
            safety_flag="warn",
            risk_level=2,
            summary=summary,
            instruction=(
                "谢谢您愿意继续说明。这些内容可能不容易谈，我需要再了解一些信息来确认您的安全。\n"
                + first.next_question
            ),
            waiting_for="bssa_answer",
        )

    def _handle_bssa(
        self, state: L1SessionState, text: str, message_id: str
    ) -> dict[str, Any]:
        core_missing_before = missing_fields(state.bssa)
        if core_missing_before:
            target = core_missing_before[0]
        else:
            safety_plan_missing_before = [
                field
                for field in missing_fields(state.bssa, include_safety_plan=True)
                if field in SAFETY_PLAN_FIELDS
            ]
            target = (
                safety_plan_missing_before[0]
                if safety_plan_missing_before
                else "clinical_clarification"
            )
        try:
            extraction = interview_bssa(text, state.bssa, target=target)
        except L1ModelFailure as exc:
            return self._error_passport(state, message_id, exc)
        apply_extraction(state.bssa, extraction)
        reason = emergency_reason(state.bssa)
        if reason:
            state.disposition = "emergency_evaluation"
            state.phase = "safety_hold"
            return self._safety_transition(
                state,
                message_id=message_id,
                disposition="emergency_evaluation",
                patient_quote=text,
                basis=reason,
            )

        core_missing = missing_fields(state.bssa)
        if core_missing:
            if state.bssa.turn_count >= 12 or state.bssa.refused_fields:
                state.disposition = "further_evaluation"
                state.phase = "safety_hold"
                return self._safety_transition(
                    state,
                    message_id=message_id,
                    disposition="further_evaluation",
                    patient_quote=text,
                    basis="BSSA关键安全信息仍不完整或患者拒绝回答。",
                )
            next_question = extraction.next_question
            if extraction.next_question_target != core_missing[0]:
                try:
                    next_turn = interview_bssa(
                        "", state.bssa, target=core_missing[0]
                    )
                except L1ModelFailure as exc:
                    return self._error_passport(state, message_id, exc)
                next_question = next_turn.next_question
            return self._passport(
                state,
                message_id=message_id,
                decision_path="safety_assessment",
                safety_flag="warn",
                risk_level=2,
                summary="BSSA正在收集必要安全信息。",
                instruction=next_question,
                waiting_for="bssa_answer",
                evidence=self._bssa_evidence(extraction.evidence_quotes),
            )

        disposition = determine_disposition(state.bssa, state.asq)
        if disposition == "further_evaluation":
            state.disposition = disposition
            state.phase = "safety_hold"
            return self._safety_transition(
                state,
                message_id=message_id,
                disposition=disposition,
                patient_quote=text,
                basis="BSSA显示仍需由专业人员进一步评估风险。",
            )
        if disposition == "non_urgent_follow_up":
            plan_missing = missing_fields(state.bssa, include_safety_plan=True)
            plan_missing = [item for item in plan_missing if item in SAFETY_PLAN_FIELDS]
            if plan_missing:
                # 若本轮刚好完成最后一个核心字段，先让模型针对安全计划的
                # 首个缺失字段生成下一问，避免沿用核心评估阶段的问题。
                next_question = extraction.next_question
                if extraction.next_question_target != plan_missing[0]:
                    try:
                        plan_turn = interview_bssa(
                            "", state.bssa, target=plan_missing[0]
                        )
                    except L1ModelFailure as exc:
                        return self._error_passport(state, message_id, exc)
                    next_question = plan_turn.next_question
                return self._passport(
                    state,
                    message_id=message_id,
                    decision_path="safety_assessment",
                    safety_flag="warn",
                    risk_level=2,
                    summary="BSSA为非紧急风险，正在完成安全计划。",
                    instruction=next_question,
                    waiting_for="safety_plan_answer",
                    evidence=self._bssa_evidence(extraction.evidence_quotes),
                )
            state.active_controls = ["monitor_suicidal_ideation"]

        state.disposition = disposition
        state.bssa.status = "completed"
        state.phase = "monitoring"
        support, source = generate_support(
            patient_quote=text,
            disposition=disposition,
            required_action=(
                "确认安全计划并安排后续随访"
                if disposition == "non_urgent_follow_up"
                else "继续普通心理评估"
            ),
        )
        state.last_support_message = support
        return self._passport(
            state,
            message_id=message_id,
            decision_path="safety_assessment",
            safety_flag="warn" if state.active_controls else "pass",
            risk_level=2 if state.active_controls else 1,
            summary=f"BSSA完成，处置为{disposition}，支持回应来源为{source}。",
            instruction=support,
            waiting_for="free_text",
            resume_l2=True,
            evidence=self._bssa_evidence(extraction.evidence_quotes),
        )

    def _safety_transition(
        self,
        state: L1SessionState,
        *,
        message_id: str,
        disposition: SafetyDisposition,
        patient_quote: str,
        basis: str,
    ) -> dict[str, Any]:
        emergency = disposition == "emergency_evaluation"
        action = (
            "普通评估已暂停。请立即联系当地急救服务或前往最近的急诊，"
            "并请可信任的人现在陪在您身边；如果可能，请先远离可能造成伤害的物品。"
            if emergency
            else
            "普通评估已暂停。请尽快联系能够进行自杀风险评估的医疗或心理健康专业人员，"
            "并让可信任的人知道您当前需要支持。"
        )
        support, source = generate_support(
            patient_quote=patient_quote,
            disposition=disposition,
            required_action=action,
        )
        state.last_support_message = support
        return self._passport(
            state,
            message_id=message_id,
            decision_path="clinical_crisis" if emergency else "safety_pause",
            safety_flag="block",
            risk_level=3 if emergency else 2,
            summary=f"{basis} 支持回应来源：{source}。",
            instruction=support + "\n\n" + action,
            evidence=[{"source": "L1_safety_assessment", "finding": patient_quote, "basis": basis}],
        )

    def _held_passport(self, state: L1SessionState, message_id: str) -> dict[str, Any]:
        emergency = state.disposition == "emergency_evaluation"
        return self._passport(
            state,
            message_id=message_id,
            decision_path="clinical_crisis" if emergency else "safety_pause",
            safety_flag="block",
            risk_level=3 if emergency else 2,
            summary="安全保持状态仍在生效。",
            instruction=(
                state.last_support_message
                + "\n\n普通疗程仍处于暂停状态，请继续按照安全页面中的步骤联系人工帮助。"
            ).strip(),
        )

    def _passport(
        self,
        state: L1SessionState,
        *,
        message_id: str,
        decision_path: str,
        safety_flag: str,
        risk_level: int,
        summary: str,
        instruction: str,
        waiting_for: str | None = None,
        resume_l2: bool = False,
        evidence: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        risk_types = ["suicidal_ideation"] if (
            decision_path in {"safety_assessment", "safety_pause", "clinical_crisis"}
            or "monitor_suicidal_ideation" in state.active_controls
        ) else []
        passport: dict[str, Any] = {
            "schema_version": "1.0",
            "layer_id": "L1",
            "session_id": state.session_id,
            "message_id": message_id,
            "timestamp": _utc_now(),
            "safety_flag": safety_flag,
            "risk_level": risk_level,
            "risk_types": risk_types,
            "crisis_flag": decision_path == "clinical_crisis",
            "decision_path": decision_path,
            "security_event": {
                "detected": decision_path == "security_reject",
                "event_type": "prompt_injection" if decision_path == "security_reject" else "",
                "severity": 3 if decision_path == "security_reject" else 0,
                "matched_patterns": (
                    state.last_signal.jailbreak_evidence if state.last_signal else []
                ),
            },
            "sensitive_topics": [],
            "summary": summary,
            "structured_labels": {
                "population": "adult_outpatient",
                "safety_signal": (
                    state.last_signal.model_dump(mode="json") if state.last_signal else {}
                ),
                "asq": {
                    **state.asq.model_dump(mode="json"),
                    "instrument": "NIMH_ASQ",
                    "language_note": "zh-CN项目翻译，非NIMH官方验证译本",
                },
                "bssa": state.bssa.model_dump(mode="json"),
                "disposition": state.disposition,
                "l2_handoff": {
                    "baseline_asq_status": state.baseline_asq_status,
                    "active_controls": state.active_controls,
                    "ordinary_care_allowed": decision_path == "normal",
                },
                "scale_skip_list": [],
            },
            "evidence_refs": evidence or [],
            "confidence": state.last_signal.confidence if state.last_signal else 1.0,
            "flags": list(state.active_controls),
            "instruction_to_next": instruction,
            "waiting_for": waiting_for,
            "resume_l2_next_turn": resume_l2,
        }
        passport["execution"] = execute_safety_policy(passport)
        passport["guard"] = self.guard.assert_valid(passport)
        return passport

    @staticmethod
    def _error_passport(
        state: L1SessionState, message_id: str, exc: L1ModelFailure
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "layer_id": "L1",
            "session_id": state.session_id,
            "message_id": message_id,
            "timestamp": _utc_now(),
            "status": "error",
            "error_code": exc.code,
            "message": "安全评估API不可用" if exc.code == "L1_API_ERROR" else "安全评估模型输出不合法",
            "terminal": True,
            "can_call_l2": False,
            "details": {"stage": exc.stage, "exception": exc.exception_type},
            "evidence_refs": [],
        }

    @staticmethod
    def _signal_evidence(signal: SafetySignalAnalysis) -> list[dict[str, str]]:
        return [
            {"source": "L1_safety_semantic_model_API", "finding": quote, "basis": signal.summary}
            for quote in signal.evidence_quotes
        ]

    @staticmethod
    def _bssa_evidence(quotes: dict[str, str]) -> list[dict[str, str]]:
        return [
            {"source": "L1_BSSA_interview_API", "finding": quote, "basis": field}
            for field, quote in quotes.items()
        ]
