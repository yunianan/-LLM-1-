"""PA6 停止条件：判断「这次自动跑该收尾了」。

要分清两个概念：
- **疗程终止**：系统侧真的结束了（危机转介、人工复核、疗程走完、患者退出），
  标志是 ``TurnResponse.session_ended`` / ``termination_reason``。
- **测试停止**：Patient Agent 侧觉得「该验证的东西都验过了」，主动停手。
  例如覆盖到了目标事件、跑满 max_turns、或守卫发现患者自己出了问题。

两者都会让运行器退出循环，但报告里区分得很清楚，避免把测试停止误当成系统行为。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from patient_agent.client.base import TurnResponse
from patient_agent.errors import ErrorCode
from patient_agent.recorder import SessionRecorder
from patient_agent.schemas import StopPolicy


class StopDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_stop: bool
    reason: str = ""
    # True 表示这是系统侧的真实终止，而不是测试主动停。
    system_terminated: bool = False


def evaluate_stop(
    policy: StopPolicy,
    *,
    turn_count: int,
    response: TurnResponse,
    recorder: SessionRecorder,
) -> StopDecision:
    """按优先级判断是否停止。越靠前的条件越「硬」。"""

    # 1) 系统侧终止：最高优先级，后面的条件都不必看了。
    if response.session_ended or response.termination_reason:
        reason = response.termination_reason or "session_ended"
        return StopDecision(
            should_stop=True,
            reason=f"系统终止：{reason}",
            system_terminated=True,
        )

    # 2) 安全暂停：L1 走了安全路径，本轮测试到此为止。
    if policy.stop_on_crisis and response.error_code == ErrorCode.SAFETY_HOLD:
        return StopDecision(
            should_stop=True,
            reason="L1 安全路径激活（SAFETY_HOLD）",
            system_terminated=True,
        )

    # 3) 人工复核。
    if policy.stop_on_human_review and response.error_code == ErrorCode.HUMAN_REVIEW:
        return StopDecision(
            should_stop=True,
            reason="系统请求人工复核（HUMAN_REVIEW）",
            system_terminated=True,
        )

    # 4) 患者侧违规：这是我们自己的问题，立刻停下以便定位。
    if policy.stop_on_violation and any(
        violation.get("blocking", True) for violation in recorder.guard_violations()
    ):
        return StopDecision(should_stop=True, reason="患者生成违反一致性约束，停止以便排查")

    # 5) 目标事件全部覆盖：验证目的已达成。
    if policy.target_events:
        covered = set(recorder.all_events())
        if all(event in covered for event in policy.target_events):
            return StopDecision(
                should_stop=True,
                reason="目标事件已全部覆盖：" + "、".join(policy.target_events),
            )

    # 6) 轮数上限：兜底，防止不收敛。
    if turn_count >= policy.max_turns:
        return StopDecision(should_stop=True, reason=f"达到 max_turns={policy.max_turns}")

    return StopDecision(should_stop=False)
