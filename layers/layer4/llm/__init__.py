from layers.layer4.config import Settings
from layers.layer4.llm.base import LLMBackend
from layers.layer4.llm.local import LocalModelBackend
from layers.layer4.llm.mock import MockBackend
from layers.layer4.llm.openai_compat import OpenAICompatBackend


def get_backend(config: Settings) -> LLMBackend:
    """按配置返回 LLM 后端实例。后期本地模型就绪时，实现 layers/layer4/llm/local.py 即可。"""
    if config.llm_provider == "mock":
        return MockBackend()
    if config.llm_provider == "openai_compat":
        return OpenAICompatBackend(
            api_key=config.llm_api_key or "",
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
    if config.llm_provider == "local":
        return LocalModelBackend()
    raise ValueError(f"未知 LLM 提供商: {config.llm_provider}")
