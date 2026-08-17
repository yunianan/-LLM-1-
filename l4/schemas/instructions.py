from typing import Literal

from pydantic import BaseModel

Substage = Literal["L4-L1", "L4-L2", "L4-L3", "L4-L4"]


class RoleInstruction(BaseModel):
    """L1 角色定义（全会话复用）。"""

    identity: str
    style: str


class DiagnosticAnchor(BaseModel):
    """L2 诊断锚点（全会话复用，由 DiagnosticReport 派生）。"""

    primary: str = ""
    comorbidities: list[str] = []
    excluded: list[str] = []
    core_beliefs: list[str] = []
    plugins: list[str] = []
    confidence: float = 0.0
    therapy_type: str = ""


class TurnInstruction(BaseModel):
    """L3 本轮动态指令（每轮重新生成）。"""

    goal: str
    technique: str
    forbidden: list[str] = []
    link_previous: str = ""
    plugin_guidance: str | None = None
    force_substage: Substage | None = None


class BoundaryInstruction(BaseModel):
    """L4 防越界指令（每轮复用）。"""

    absolute_bans: list[str] = []
    safety_trigger: str = ""


class SafetyPassport(BaseModel):
    """L1 安全通行证中 L4 需要的部分（每轮复用）。"""

    risk_level: int = 1
    sensitive_topics: list[str] = []
    flags: list[str] = []
