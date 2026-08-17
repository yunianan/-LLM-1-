from l4.llm.base import LLMBackend, Message

DEFAULT_MOCK_RESPONSE = (
    '{"reply": "我听到你说的了，这很不容易。我们可以一起看看现在的情况。"'
    ', "stage_complete": false, "reason": "mock 默认回复"}'
)


class MockBackend(LLMBackend):
    """FakeLLM：按最后一条用户消息子串匹配预设回复。无密钥可跑测试与演示。"""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default or DEFAULT_MOCK_RESPONSE

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        for key, value in self._responses.items():
            if key and key in last_user:
                return value
        return self._default
