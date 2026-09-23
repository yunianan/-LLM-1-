from typing import Literal

from pydantic import BaseModel, model_validator

Verdict = Literal["pass", "fail"]


class ReviewResult(BaseModel):
    """L5 审核结果（Mock 审核器与真实 L5 共用）。"""

    harm: Verdict
    boundary: Verdict
    quality: Verdict
    fail_reason: str = ""
    fix_instruction: str = ""

    @model_validator(mode="after")
    def reject_contradictory_pass(self) -> "ReviewResult":
        self.fail_reason = self.fail_reason.strip()
        self.fix_instruction = self.fix_instruction.strip()
        if (self.harm == self.boundary == self.quality == "pass"
                and (self.fail_reason or self.fix_instruction)):
            # 无法从矛盾结果确认哪一维可靠，整次审核按不通过处理。
            self.harm = self.boundary = self.quality = "fail"
            self.fail_reason = "审核结果矛盾：三项通过但仍包含失败原因或修复指令。" + self.fail_reason
            self.fix_instruction = self.fix_instruction or "依据审核指出的问题重写回复，并重新完成三项审核。"
        return self

    @property
    def passed(self) -> bool:
        return (self.harm == self.boundary == self.quality == "pass"
                and not self.fail_reason.strip() and not self.fix_instruction.strip())
