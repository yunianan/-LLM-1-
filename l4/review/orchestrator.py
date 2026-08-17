from collections.abc import Callable

from pydantic import BaseModel

from l4.generation.responder import Responder
from l4.llm.base import Message
from l4.review.reviewer import Reviewer
from l4.schemas.instructions import TurnInstruction
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult

DEFAULT_FALLBACK = "我可能需要重新整理一下思路，我们能换个角度聊吗？"


class ReviewOutcome(BaseModel):
    reply: str
    review_history: list[ReviewResult]
    attempts: int
    fallback_used: bool
    fix_instructions: list[str]
    stage_complete: bool = False


class ReviewOrchestrator:
    """回环A 编排：生成 → 审核 → 注入 fix_instruction 重写，最多 max_attempts 轮。"""

    def __init__(
        self,
        responder: Responder,
        reviewer: Reviewer,
        max_attempts: int = 3,
        fallback_reply: str = DEFAULT_FALLBACK,
    ) -> None:
        self._responder = responder
        self._reviewer = reviewer
        self._max_attempts = max_attempts
        self._fallback_reply = fallback_reply

    def run(
        self,
        turn: TurnInstruction,
        history: list[Message],
        user_message: str,
        build_ctx: Callable[[str], L5Context],
        validate: Callable[[str], list[str]] | None = None,
        initial_fix: str | None = None,
    ) -> ReviewOutcome:
        fix: str | None = initial_fix
        history_reviews: list[ReviewResult] = []
        fix_instructions: list[str] = []
        last_stage_complete = False

        for _ in range(self._max_attempts):
            gen = self._responder.generate(
                turn, history, user_message, fix_instruction=fix
            )
            violations = validate(gen.reply) if validate else []
            if violations:
                result = ReviewResult(
                    harm="pass", boundary="pass", quality="fail",
                    fail_reason="L4本地校验未通过",
                    fix_instruction="请修正以下问题：" + "；".join(violations),
                )
            else:
                result = self._reviewer.review(build_ctx(gen.reply), history_reviews)
            history_reviews.append(result)
            if result.fix_instruction:
                fix_instructions.append(result.fix_instruction)
            last_stage_complete = gen.stage_complete
            if result.passed:
                return ReviewOutcome(
                    reply=gen.reply,
                    review_history=history_reviews,
                    attempts=len(history_reviews),
                    fallback_used=False,
                    fix_instructions=fix_instructions,
                    stage_complete=last_stage_complete,
                )
            fix = result.fix_instruction

        return ReviewOutcome(
            reply=self._fallback_reply,
            review_history=history_reviews,
            attempts=len(history_reviews),
            fallback_used=True,
            fix_instructions=fix_instructions,
            stage_complete=False,
        )
