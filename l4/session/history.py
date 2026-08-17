from typing import Literal

from pydantic import BaseModel

from l4.llm.base import Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.state_machine import SubstageMachine


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionData(BaseModel):
    session_id: str
    machine: SubstageMachine = SubstageMachine()
    history: list[ConversationTurn] = []
    l3_history: list[TurnInstruction] = []
    role: RoleInstruction = RoleInstruction(identity="", style="")
    anchor: DiagnosticAnchor | None = None
    boundary: BoundaryInstruction = BoundaryInstruction()
    safety: SafetyPassport = SafetyPassport()
    therapy_options: list[str] = []
    meta: dict = {}

    def get_messages(self) -> list[Message]:
        return [Message(role=t.role, content=t.content) for t in self.history]


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
