from datetime import datetime, timezone

from layers.layer4.schemas.instructions import (
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from layers.layer4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from layers.layer4.schemas.summary import EvidenceRef, StructuredSummary
from layers.layer4.session.history import ConversationTurn
from shared.schemas.treatment_instructions import TreatmentInstructions


def build_l5_context(
    reply: str,
    substage: Substage,
    safety: SafetyPassport,
    instructions: TreatmentInstructions,
    user_message: str,
    history: list[ConversationTurn],
) -> L5Context:
    anchor = instructions.L2
    suicidal = "passive_only" if "monitor_suicidal_ideation" in safety.flags else None
    return L5Context(
        l4_raw_output=reply,
        l4_substage=substage,
        therapy_type=anchor.therapy_type,
        user_message=user_message,
        history=history,
        instructions=instructions,
        diagnosis_context=DiagnosisContext(
            primary=anchor.primary,
            secondary=anchor.comorbidities,
            suicidal_ideation=suicidal,
        ),
        risk_context=RiskContext(
            risk_level=safety.risk_level,
            sensitive_topics=safety.sensitive_topics,
            flags=safety.flags,
        ),
    ).model_copy(deep=True)


def build_evidence_refs(
    turn: TurnInstruction,
    history: list[ConversationTurn],
    safety: SafetyPassport,
) -> list[EvidenceRef]:
    refs = [
        EvidenceRef(
            source="L3_turn_instruction",
            finding=f"technique={turn.technique}",
            basis=f"本轮目标：{turn.goal}",
        )
    ]
    if history:
        last_user = next(
            (t.content for t in reversed(history) if t.role == "user"), ""
        )
        refs.append(
            EvidenceRef(
                source="conversation_history",
                finding=f"{len(history)} 条历史消息",
                basis=f"最近用户消息：{last_user[:50]}",
            )
        )
    if safety.flags:
        refs.append(
            EvidenceRef(
                source="L1_safety_passport",
                finding=f"risk_level={safety.risk_level}",
                basis="；".join(safety.flags),
            )
        )
    return refs


def build_summary(
    turn: TurnInstruction,
    substage: Substage,
    reply: str,
    evidence_refs: list[EvidenceRef],
    confidence: float = 0.85,
    flags: list[str] | None = None,
) -> StructuredSummary:
    return StructuredSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary=reply[:200],
        structured_labels={
            "goal": turn.goal,
            "technique": turn.technique,
            "substage": substage,
            "plugin_guidance": turn.plugin_guidance or "",
        },
        evidence_refs=evidence_refs,
        confidence=confidence,
        flags=list(flags or []),
    )
