from typing import Literal

from pydantic import BaseModel

Verdict = Literal["pass", "fail"]


class ReviewResult(BaseModel):
    """L5 审核结果（Mock 审核器与真实 L5 共用）。"""

    harm: Verdict
    boundary: Verdict
    quality: Verdict
    fail_reason: str = ""
    fix_instruction: str = ""

    @property
    def passed(self) -> bool:
        return self.harm == "pass" and self.boundary == "pass" and self.quality == "pass"
