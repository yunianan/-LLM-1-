from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """L4 治疗对话层配置。环境变量前缀 L4_，例：L4_LLM_PROVIDER=mock。"""

    model_config = SettingsConfigDict(env_prefix="L4_", extra="ignore")

    llm_provider: Literal["openai_compat", "mock", "local"] = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7

    review_mode: Literal["mock", "l5_endpoint"] = "mock"
    max_rewrite_attempts: int = 3
    fallback_reply: str = "我可能需要重新整理一下思路，我们能换个角度聊吗？"
