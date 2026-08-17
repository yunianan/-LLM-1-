import pytest

from l4.config import Settings
from l4.llm import get_backend
from l4.llm.base import Message
from l4.llm.local import LocalModelBackend
from l4.llm.mock import MockBackend
from l4.llm.openai_compat import OpenAICompatBackend


def test_mock_json_mode_matches_substring():
    backend = MockBackend(
        responses={"触发词": '{"reply": "违规回复", "stage_complete": false, "reason": "x"}'},
        default='{"reply": "正常回复", "stage_complete": true, "reason": "y"}',
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="这里包含触发词，请回复"),
    ]
    assert backend.chat(messages, json_mode=True) == '{"reply": "违规回复", "stage_complete": false, "reason": "x"}'

    other = [Message(role="user", content="没有关键词")]
    assert backend.chat(other, json_mode=True) == '{"reply": "正常回复", "stage_complete": true, "reason": "y"}'


def test_mock_non_json_returns_raw():
    backend = MockBackend(default="普通文本回复")
    out = backend.chat([Message(role="user", content="hi")], json_mode=False)
    assert out == "普通文本回复"


def test_factory_dispatch():
    assert isinstance(get_backend(Settings(_env_file=None, llm_provider="mock")), MockBackend)
    assert isinstance(
        get_backend(Settings(_env_file=None, llm_provider="openai_compat",
                             llm_api_key="sk-x", llm_base_url="https://api.deepseek.com",
                             llm_model="deepseek-chat")),
        OpenAICompatBackend,
    )


def test_factory_local_raises():
    with pytest.raises(NotImplementedError):
        get_backend(Settings(_env_file=None, llm_provider="local"))
