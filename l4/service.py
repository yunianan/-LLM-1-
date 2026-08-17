import uuid

from pydantic import BaseModel

from l4.config import Settings
from l4.generation.responder import Responder
from l4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)
from l4.llm.base import LLMBackend
from l4.llm import get_backend
from l4.review.mock_reviewer import MockReviewer
from l4.review.orchestrator import ReviewOrchestrator
from l4.review.reviewer import Reviewer
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult
from l4.schemas.summary import StructuredSummary
from l4.session.history import ConversationTurn, SessionData, SessionStore

THERAPY_KEYWORDS = {
    "EMDR": "EMDR", "CPT": "CPT", "IPSRT": "IPSRT", "ERP": "ERP",
    "CBT-I": "CBT-I", "ACT": "ACT", "MBCT": "MBCT", "IPT": "IPT",
    "SFBT": "SFBT", "CBT": "CBT",
}


class SessionRequest(BaseModel):
    session_id: str | None = None
    role: RoleInstruction
    report: DiagnosticReport
    boundary: BoundaryInstruction = BoundaryInstruction()
    safety: SafetyPassport = SafetyPassport()


class TurnRequest(BaseModel):
    session_id: str
    user_message: str
    turn: TurnInstruction


class TurnOutcome(BaseModel):
    session_id: str
    reply: str
    l4_substage: Substage
    l5_context: L5Context
    summary: StructuredSummary
    review_history: list[ReviewResult]
    fallback_used: bool = False


class L4Service:
    """L4 治疗对话层门面：一轮完整处理（输入指令 → 回复 + L5Context + 摘要）。"""

    def __init__(
        self,
        config: Settings | None = None,
        backend: LLMBackend | None = None,
        reviewer: Reviewer | None = None,
    ) -> None:
        self.config = config or Settings()
        self.backend = backend or get_backend(self.config)
        if self.config.review_mode == "l5_endpoint":
            raise NotImplementedError(
                "review_mode='l5_endpoint' 尚未实现，请使用 'mock'，或真实 L5 就绪后注入 reviewer 实现"
            )
        self.reviewer = reviewer or MockReviewer()
        self.store = SessionStore()

    def create_session(self, req: SessionRequest) -> str:
        session_id = req.session_id or uuid.uuid4().hex
        anchor = req.report.to_anchor()
        therapy_options = [t for t in [req.report.recommended_therapy.primary]
                           + list(req.report.recommended_therapy.addon) if t]
        data = SessionData(
            session_id=session_id,
            role=req.role,
            anchor=anchor,
            boundary=req.boundary,
            safety=req.safety,
            therapy_options=therapy_options,
        )
        self.store.create(data)
        return session_id

    def handle_turn(self, req: TurnRequest, initial_fix: str | None = None) -> TurnOutcome:
        sd = self.store.get(req.session_id)
        anchor = sd.anchor or DiagnosticAnchor()
        substage = sd.machine.resolve(req.turn.force_substage)

        responder = Responder(self.backend, sd.role, anchor, sd.boundary)
        orchestrator = ReviewOrchestrator(
            responder, self.reviewer,
            self.config.max_rewrite_attempts, self.config.fallback_reply,
        )

        def build_ctx(reply: str) -> L5Context:
            return build_l5_context(reply, substage, anchor.therapy_type, sd.safety, anchor)

        def validate(reply: str) -> list[str]:
            return self._validate(reply, req.turn, sd.therapy_options)

        outcome = orchestrator.run(
            req.turn, sd.history, req.user_message,
            build_ctx, validate, initial_fix=initial_fix,
        )

        flags: list[str] = list(sd.machine.flags)
        if outcome.fallback_used:
            flags.extend(["loop_A_fallback", "manual_review_required"])

        summary = build_summary(
            req.turn, substage, outcome.reply,
            build_evidence_refs(req.turn, sd.history, sd.safety),
            flags=flags,
        )

        sd.history.append(ConversationTurn(role="user", content=req.user_message))
        sd.history.append(ConversationTurn(role="assistant", content=outcome.reply))
        sd.l3_history.append(req.turn)
        if not outcome.fallback_used:
            sd.machine.advance(stage_complete=outcome.stage_complete,
                               force=req.turn.force_substage)

        return TurnOutcome(
            session_id=req.session_id,
            reply=outcome.reply,
            l4_substage=substage,
            l5_context=build_ctx(outcome.reply),
            summary=summary,
            review_history=outcome.review_history,
            fallback_used=outcome.fallback_used,
        )

    @staticmethod
    def _validate(reply: str, turn: TurnInstruction,
                  therapy_options: list[str]) -> list[str]:
        """V4 本地自校验：禁止词 + V4-3 疗法一致性（启发式）。"""
        violations: list[str] = []
        for word in turn.forbidden:
            if word and word in reply:
                violations.append(f"回复包含禁止词：{word}")
        allowed = set(therapy_options)
        for kw in THERAPY_KEYWORDS:
            if kw in reply and kw not in allowed and kw != "CBT":
                violations.append(f"回复提及未授权疗法：{kw}")
        return violations
