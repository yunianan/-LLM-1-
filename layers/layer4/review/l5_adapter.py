from collections.abc import Callable

from layers.layer4.schemas.l5_context import L5Context
from layers.layer4.schemas.review import ReviewResult


class L5Reviewer:
    """将外部 L5 审核服务包装为 L4 Reviewer 协议。

    review_fn: Callable[[dict], dict]，接收 ContextualSafetyContext 字典，
    返回文档格式的 L5 审核结果 {harm, boundary, quality, fail_reason, fix_instruction, ...}。
    """

    def __init__(self, review_fn: Callable[[dict], dict]) -> None:
        self._review_fn = review_fn

    def review(self, ctx: L5Context, review_history: list[ReviewResult]) -> ReviewResult:
        payload = ctx.model_dump(exclude_none=True)
        try:
            data = self._review_fn(payload)
            return ReviewResult(
                harm=data["harm"],
                boundary=data["boundary"],
                quality=data["quality"],
                fail_reason=data.get("fail_reason", ""),
                fix_instruction=data.get("fix_instruction", ""),
            )
        except Exception as exc:
            return ReviewResult(
                harm="fail",
                boundary="fail",
                quality="fail",
                fail_reason=f"L5审核调用失败：{type(exc).__name__}",
                fix_instruction=(
                    "L5审核暂不可用。不得放行当前回复；请生成更保守、非诊断性、"
                    "不包含医疗建议的安全回复。"
                ),
            )
