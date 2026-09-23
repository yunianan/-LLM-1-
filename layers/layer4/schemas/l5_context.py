from pydantic import BaseModel, Field, model_validator

from layers.layer4.schemas.instructions import Substage
from layers.layer4.session.history import ConversationTurn
from shared.schemas.treatment_instructions import TreatmentInstructions


class DiagnosisContext(BaseModel):
    primary: str = ""
    secondary: list[str] = []
    suicidal_ideation: str | None = None


class RiskContext(BaseModel):
    risk_level: int = 1
    sensitive_topics: list[str] = []
    flags: list[str] = []


class L5Context(BaseModel):
    """接口四：L4 + L3 → L5 ContextualSafetyContext（上下文安全上下文）。"""

    l4_raw_output: str
    l4_substage: Substage
    therapy_type: str
    user_message: str = Field(min_length=1)
    history: list[ConversationTurn]
    instructions: TreatmentInstructions
    diagnosis_context: DiagnosisContext = DiagnosisContext()
    risk_context: RiskContext = RiskContext()

    @model_validator(mode="after")
    def require_turn_instruction(self) -> "L5Context":
        if self.instructions.L3 is None:
            raise ValueError("L5 审核必须包含本轮 L3 治疗指令")
        return self
