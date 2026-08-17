from typing import Literal

from pydantic import BaseModel

from l4.schemas.instructions import Substage

SUBSTAGE_ORDER: tuple[Substage, ...] = ("L4-L1", "L4-L2", "L4-L3", "L4-L4")


class SubstageMachine(BaseModel):
    """L4 子层状态机：L4-L1 共情 → L4-L2 结果反馈 → L4-L3 技术执行 → L4-L4 作业布置。

    推进规则（三路信号取最高优先级）：
    1. force（L3 强制覆盖）非空 → 直接跳转，回退时打 stage_regression_override；
    2. stage_complete=True（阶段完成判定）→ 前进一步；
    3. 其余情况保持当前阶段。
    """

    current: Substage = "L4-L1"
    flags: list[str] = []

    def resolve(self, force: Substage | None = None) -> Substage:
        """返回本轮应使用的子阶段（不推进）。"""
        return force or self.current

    def advance(self, stage_complete: bool = False, force: Substage | None = None) -> None:
        if force is not None:
            if SUBSTAGE_ORDER.index(force) < SUBSTAGE_ORDER.index(self.current):
                self._flag("stage_regression_override")
            self.current = force
            return
        idx = SUBSTAGE_ORDER.index(self.current)
        if stage_complete and idx < len(SUBSTAGE_ORDER) - 1:
            self.current = SUBSTAGE_ORDER[idx + 1]
        elif stage_complete and idx == len(SUBSTAGE_ORDER) - 1:
            self._flag("terminal_reached")

    def _flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
