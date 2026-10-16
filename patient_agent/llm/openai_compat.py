"""OpenAI 兼容接口适配器（``/chat/completions``）。

只依赖标准库 ``urllib``。``base_url`` 可指向 OpenAI、也可指向自建/第三方网关，
只要它实现 ``POST {base_url}/chat/completions`` 就可用。

失败语义：
- 超时 → ``LLMError``（error_code=TIMEOUT）
- HTTP 错误 / 连接失败 → ``LLMError``（error_code=API_ERROR）
- 返回体没有 choices → ``LLMError``（error_code=SCHEMA_ERROR）
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 6】真实大模型 —— 患者台词的「嘴」
# 接入什么：任意 OpenAI 兼容的 /chat/completions 服务（OpenAI / DeepSeek / 通义千问 /
#           自建网关 / 本地 vLLM，示例见 .env.example）。生成器只把「这一轮要说什么」
#           写进 prompt，真正把它说成人话的是这个适配器。
# 谁提供：本地或云端 LLM 服务；需要 base_url 与 api_key。
# 怎么打开：PATIENT_AGENT_LLM_MODE=auto 或 api，并填 OPENAI_BASE_URL / OPENAI_API_KEY /
#           OPENAI_MODEL（开关见 llm/__init__.py【接入点 7】、config.py【接入点 9】）。
# 换供应商：通常只改 OPENAI_BASE_URL，只要它实现 POST {base_url}/chat/completions。
# 换协议（Anthropic Messages、通义原生等）时才需要新增适配器，在 llm/__init__.py 加分支。
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from patient_agent.errors import ErrorCode, LLMError
from patient_agent.llm.base import LLMClient


class OpenAICompatLLM(LLMClient):
    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.is_available():
            raise LLMError(
                "OpenAI 兼容适配器缺少 api_key 或 base_url",
                error_code=ErrorCode.API_ERROR,
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": (
                self.default_temperature if temperature is None else temperature
            ),
        }
        token_limit = self.default_max_tokens if max_tokens is None else max_tokens
        if token_limit is not None:
            payload["max_tokens"] = token_limit

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except socket.timeout as exc:
            raise LLMError(
                f"LLM 调用超时（{self.timeout}s）",
                error_code=ErrorCode.TIMEOUT,
                detail=str(exc),
            ) from exc
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover
                detail = ""
            raise LLMError(
                f"LLM 调用失败：HTTP {exc.code}",
                error_code=ErrorCode.API_ERROR,
                detail=detail,
            ) from exc
        except urllib.error.URLError as exc:
            code = ErrorCode.TIMEOUT if isinstance(exc.reason, socket.timeout) else ErrorCode.API_ERROR
            raise LLMError(
                f"无法连接 LLM 服务：{exc.reason}",
                error_code=code,
                detail=str(exc.reason),
            ) from exc

        try:
            parsed = json.loads(raw)
            content = parsed["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "LLM 返回体缺少 choices[0].message.content",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail=raw[:500],
            ) from exc
        if not isinstance(content, str):
            raise LLMError(
                "LLM 返回的 content 不是字符串",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail=str(content)[:200],
            )
        return content.strip()
