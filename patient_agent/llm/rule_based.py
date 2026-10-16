"""离线确定性「伪模型」。

不是把规则藏在生成器里，而是把生成器已经算好的候选回答，放在 prompt 的
``<output>...</output>`` 块里，让这个适配器原样回显。这样做的好处：

- 生成器里的 prompt 模板、上下文组装、字段顺序，与接真模型时**完全一致**；
- 没有网络与随机性，pytest 与 CI 能在任何机器上得到同一结果；
- 想切真模型，只换适配器，生成器一行不改。

回显前会剥掉 ``<output>`` 标签本身，所以上层拿到的是干净的患者台词。
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 8】离线伪模型（真实模型的对照组）
# 接入什么：本文件不是被接入的对象。接上真模型之后，offline 模式与 CI 仍然用它。
# 说明：它回显 prompt 里的 <output> 块，因此能证明「生成器逻辑与是否用真模型无关」——
#       接真模型时生成器一行不改，只换适配器。
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re

from patient_agent.errors import LLMError
from patient_agent.llm.base import LLMClient

_OUTPUT_BLOCK_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)


class RuleBasedLLM(LLMClient):
    """从 prompt 里提取预置输出，模拟一次「模型调用」。"""

    name = "rule_based"

    def __init__(self, *, strict: bool = True) -> None:
        # strict=True 时找不到 <output> 块就报错，避免静默产出空台词；
        # 测试里想验证「缺块」行为可置 False。
        self.strict = strict
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"system": system, "user": user})
        match = _OUTPUT_BLOCK_RE.search(user)
        if match is None:
            if self.strict:
                raise LLMError("离线适配器的 prompt 缺少 <output> 块，无法回显")
            return ""
        return match.group(1).strip()

    def is_available(self) -> bool:
        return True
