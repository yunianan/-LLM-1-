import json

import pytest

from layers.layer2.llm import client, debug_output
from tests.test_l2_model_failure import _invoke_api_model, _valid_generation_response


@pytest.fixture
def debug_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(debug_output, "DEBUG_DIR", tmp_path / "l2-debug")
    monkeypatch.setenv("L2_DEBUG_OUTPUT", "true")
    return debug_output.DEBUG_DIR


def _respond(monkeypatch, raw, finish_reason="stop"):
    def response(*args, **kwargs):
        if "response_metadata" in kwargs:
            kwargs["response_metadata"]["finish_reason"] = finish_reason
        return raw
    monkeypatch.setattr(client, "ask_llm", response)


def _meta(debug_dir):
    paths = list(debug_dir.glob("*.meta.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_debug_off_by_default_writes_nothing(monkeypatch, debug_dir):
    monkeypatch.delenv("L2_DEBUG_OUTPUT")
    _respond(monkeypatch, "invalid")
    with pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    assert not debug_dir.exists()


@pytest.mark.parametrize("raw", ["", '{"reply":"private response"', '说明文字\n{"reply":"test"}'])
def test_invalid_json_records_location_and_exact_response(monkeypatch, debug_dir, capsys, raw):
    _respond(monkeypatch, raw, "length")
    with pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    info = _meta(debug_dir)
    assert info["stage"] == "L2.generate.json_parse"
    assert info["finish_reason"] == "length"
    assert info["parse_success"] is False
    assert info["content_length"] == len(raw)
    assert info["json_error"]["line"] >= 1
    assert info["json_error"]["column"] >= 1
    assert (debug_dir / info["response_file"]).read_text(encoding="utf-8") == raw
    terminal = capsys.readouterr().out
    assert "private response" not in terminal
    assert ".meta.json" in terminal


@pytest.mark.parametrize("fenced", [False, True])
def test_valid_output_records_only_metadata(monkeypatch, debug_dir, fenced):
    raw = _valid_generation_response(reply="private successful reply")
    if fenced:
        raw = "```json\n" + raw + "\n```"
    _respond(monkeypatch, raw)
    assert _invoke_api_model("generate") == "private successful reply"
    info = _meta(debug_dir)
    assert info["parse_success"] is True and info["output_valid"] is True
    assert "private successful reply" not in json.dumps(info)
    assert not list(debug_dir.glob("*.response.txt"))


def test_schema_failure_does_not_save_raw_response(monkeypatch, debug_dir):
    _respond(monkeypatch, '{"reply":"private reply"}')
    with pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    info = _meta(debug_dir)
    assert info["parse_success"] is True and info["output_valid"] is False
    assert info["stage"] == "L2.generate.schema_validation"
    assert not list(debug_dir.glob("*.response.txt"))


def test_api_failure_does_not_record_exception_message(monkeypatch, debug_dir):
    def fail(*args, **kwargs):
        raise RuntimeError("private error including secret key")
    monkeypatch.setattr(client, "ask_llm", fail)
    with pytest.raises(client.L2ModelUnavailable):
        _invoke_api_model("generate")
    info = _meta(debug_dir)
    assert info["stage"] == "L2.generate.api_request"
    assert info["parse_success"] is None
    assert info["content_length"] is None
    assert "private" not in json.dumps(info)
    assert not list(debug_dir.glob("*.response.txt"))


def test_disk_failure_preserves_successful_reply(monkeypatch, debug_dir, caplog):
    debug_dir.write_text("not a directory", encoding="utf-8")
    _respond(monkeypatch, _valid_generation_response())
    assert _invoke_api_model("generate")
    assert "调试记录写入失败" in caplog.text
