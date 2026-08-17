import json

from pydantic import BaseModel, ValidationError

from l4.generation.prompt_assembler import assemble_prompt
from l4.llm.base import LLMBackend, Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    TurnInstruction,
)


class GenerationResult(BaseModel):
    reply: str
    stage_complete: bool = False
    reason: str = ""


class Responder:
    """调用 LLM 生成治疗回复并解析输出。"""

    def __init__(
        self,
        backend: LLMBackend,
        role: RoleInstruction,
        anchor: DiagnosticAnchor,
        boundary: BoundaryInstruction,
    ) -> None:
        self._backend = backend
        self._role = role
        self._anchor = anchor
        self._boundary = boundary

    def generate(
        self,
        turn: TurnInstruction,
        history: list[Message],
        user_message: str,
        fix_instruction: str | None = None,
    ) -> GenerationResult:
        messages = assemble_prompt(
            role=self._role, anchor=self._anchor, boundary=self._boundary,
            turn=turn, history=history, user_message=user_message,
            fix_instruction=fix_instruction,
        )
        raw = self._backend.chat(messages, json_mode=True)
        try:
            parsed = json.loads(raw)
            return GenerationResult.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, TypeError):
            fallback = self._backend.chat(messages, json_mode=False)
            return GenerationResult(reply=fallback, stage_complete=False,
                                    reason="parse_fallback")
