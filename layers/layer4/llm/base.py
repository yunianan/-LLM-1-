from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


@runtime_checkable
class LLMBackend(Protocol):
    """LLM 后端端口。所有适配器（API/Mock/本地模型）实现此协议。"""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """messages 为 [{role, content}] 消息列表；json_mode=True 时要求返回合法 JSON 文本。"""
        ...


class LLMError(Exception):
    pass
