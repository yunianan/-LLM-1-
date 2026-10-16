"""LLM 适配器抽象。

生成器（PA3）只依赖这个接口，因此「离线确定性」和「真实模型」是同一个调用点：
- ``RuleBasedLLM`` 从 prompt 里的 ``<output>...</output>`` 块回显，零依赖、可复现；
- ``OpenAICompatLLM`` 走 OpenAI 兼容 HTTP 接口，``base_url`` 可指向自建网关。

两者接收完全一样的 prompt，所以离线跑通的 prompt 逻辑，接上真模型不用改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """最小可用的大模型客户端接口。"""

    name: str = "llm"

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """返回模型生成的文本（已去掉首尾空白）。失败时抛 ``LLMError``。"""

    def is_available(self) -> bool:
        """离线适配器恒为 True；需要网络的适配器在缺 key/地址时返回 False。"""

        return True
