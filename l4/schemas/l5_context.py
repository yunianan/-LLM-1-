from pydantic import BaseModel

from l4.schemas.instructions import Substage


class DiagnosisContext(BaseModel):
    primary: str = ""
    secondary: list[str] = []
    suicidal_ideation: str | None = None


class RiskContext(BaseModel):
    risk_level: int = 1
    sensitive_topics: list[str] = []


class L5Context(BaseModel):
    """接口四：L4 + L3 → L5 ContextualSafetyContext（上下文安全上下文）。"""

    l4_raw_output: str
    l4_substage: Substage
    therapy_type: str
    diagnosis_context: DiagnosisContext = DiagnosisContext()
    risk_context: RiskContext = RiskContext()
