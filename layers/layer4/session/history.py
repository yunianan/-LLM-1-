from typing import Literal

from pydantic import BaseModel

from layers.layer4.llm.base import Message
from layers.layer4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from layers.layer4.session.state_machine import SubstageMachine
from shared.schemas.treatment_instructions import TreatmentInstructions, TreatmentTurnRecord


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionData(BaseModel):
    session_id: str
    machine: SubstageMachine = SubstageMachine()
    history: list[ConversationTurn] = []
    l3_history: list[TurnInstruction] = []
    fallback_history: list[bool] = []
    manual_review_required: bool = False
    role: RoleInstruction = RoleInstruction(identity="", style="")
    anchor: DiagnosticAnchor | None = None
    boundary: BoundaryInstruction = BoundaryInstruction()
    safety: SafetyPassport = SafetyPassport()
    therapy_options: list[str] = []
    meta: dict = {}

    def get_messages(self) -> list[Message]:
        return [Message(role=t.role, content=t.content) for t in self.history]

    def instructions(self) -> TreatmentInstructions:
        return TreatmentInstructions(L1=self.role, L2=self.anchor, L4=self.boundary)

    def treatment_records(self) -> list[TreatmentTurnRecord]:
        return [TreatmentTurnRecord(
            instruction=turn,
            user_message=self.history[2 * index].content,
            reply=self.history[2 * index + 1].content,
            fallback_used=self.fallback_history[index],
        ) for index, turn in enumerate(self.l3_history)]


class SessionStore:
    """内存会话存储：按 session_id 维护 SessionData。"""

    def __init__(self) -> None:
        self._data: dict[str, SessionData] = {}

    def create(self, data: SessionData) -> None:
        self._data[data.session_id] = data

    def get(self, session_id: str) -> SessionData:
        return self._data[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._data
