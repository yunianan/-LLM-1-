from typing import Protocol, runtime_checkable

from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult


@runtime_checkable
class Reviewer(Protocol):
    """审核端口：真实 L5 与 Mock 审核器共用此协议。"""

    def review(
        self,
        ctx: L5Context,
        review_history: list[ReviewResult],
    ) -> ReviewResult:
        """审核一条 L4 输出，返回 harm/boundary/quality 判定。"""
        ...
