import os

import pytest

from l4.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.review_mode == "mock"
    assert s.llm_model == "deepseek-chat"
    assert s.max_rewrite_attempts == 3
    assert s.fallback_reply == "我可能需要重新整理一下思路，我们能换个角度聊吗？"


def test_env_override(monkeypatch):
    monkeypatch.setenv("L4_LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("L4_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("L4_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("L4_MAX_REWRITE_ATTEMPTS", "5")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai_compat"
    assert s.llm_base_url == "https://api.deepseek.com"
    assert s.llm_api_key == "sk-test"
    assert s.max_rewrite_attempts == 5


def test_invalid_provider():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="nope")
