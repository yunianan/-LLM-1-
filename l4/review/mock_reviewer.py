from l4.review.reviewer import Reviewer
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult


class MockReviewer(Reviewer):
    """可脚本化审核器：默认全通过；fail_first 前 N 次失败；fail_trigger 命中失败；always_fail 恒失败。"""

    def __init__(
        self,
        fail_first: int = 0,
        always_fail: bool = False,
        fail_trigger: str = "",
    ) -> None:
        self._fail_first = fail_first
        self._always_fail = always_fail
        self._fail_trigger = fail_trigger

    def review(
        self,
        ctx: L5Context,
        review_history: list[ReviewResult],
    ) -> ReviewResult:
        failed = (
            self._always_fail
            or len(review_history) < self._fail_first
            or (self._fail_trigger and self._fail_trigger in ctx.l4_raw_output)
        )
        if not failed:
            return ReviewResult(harm="pass", boundary="pass", quality="pass")
        return ReviewResult(
            harm="fail",
            boundary="fail",
            quality="fail",
            fail_reason=(
                f"Mock审核未通过（触发：{self._fail_trigger or '脚本配置'}，"
                f"已尝试 {len(review_history) + 1} 次）"
            ),
            fix_instruction="请改写回复：删除违规内容，使用协作式、非确定性的措辞，并保留共情。",
        )
