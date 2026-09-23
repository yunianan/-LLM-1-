from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from shared.llm_client import get_base_url, get_model


class Settings(BaseSettings):
    """L4 治疗对话层配置；生产默认使用 OpenAI-compatible API。"""

    model_config = SettingsConfigDict(env_prefix="L4_", extra="ignore")

    llm_provider: Literal["openai_compat", "mock", "local"] = "openai_compat"
    llm_base_url: str | None = Field(default_factory=get_base_url)
    llm_api_key: str | None = None
    llm_model: str = Field(default_factory=get_model)
    llm_temperature: float = 0.7

    review_mode: Literal["l5_local", "mock", "l5_endpoint"] = "l5_local"
    max_rewrite_attempts: int = 3
    fallback_reply: str = "我可能需要重新整理一下思路，我们能换个角度聊吗？"
