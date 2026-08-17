from openai import OpenAI

from l4.llm.base import LLMBackend, Message


class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容接口适配器（DeepSeek 等）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        payload: list[dict[str, str]] = [m.model_dump() for m in messages]
        kwargs: dict = {"model": self._model, "messages": payload, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content or ""
