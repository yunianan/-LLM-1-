import json
from pathlib import Path

from fastapi.testclient import TestClient

from l4.api.server import create_app
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    TurnInstruction,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

ROLE = RoleInstruction(identity="CBT治疗师", style="温暖")
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药"])


def _client():
    return TestClient(create_app())


def _session_payload(session_id=None):
    report = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    payload = {"role": ROLE.model_dump(), "report": report,
               "boundary": BOUNDARY.model_dump()}
    if session_id:
        payload["session_id"] = session_id
    return payload


def test_healthz():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_session_and_turn_flow():
    client = _client()
    resp = client.post("/v1/l4/session", json=_session_payload("s-api"))
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "s-api"

    turn = TurnInstruction(goal="建立共情", technique="开放式提问",
                           force_substage="L4-L1")
    resp = client.post("/v1/l4/turn", json={
        "session_id": "s-api",
        "user_message": "我最近压力很大，睡不好。",
        "turn": turn.model_dump(),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["l4_substage"] == "L4-L1"
    assert body["reply"]
    assert body["l5_context"]["l4_substage"] == "L4-L1"
    assert body["summary"]["layer_id"] == "L4"


def test_turn_unknown_session_404():
    resp = _client().post("/v1/l4/turn", json={
        "session_id": "nope",
        "user_message": "hi",
        "turn": TurnInstruction(goal="g", technique="t").model_dump(),
    })
    assert resp.status_code == 404


def test_rewrite_endpoint():
    client = _client()
    client.post("/v1/l4/session", json=_session_payload("s-rw"))
    resp = client.post("/v1/l4/rewrite", json={
        "session_id": "s-rw",
        "user_message": "hi",
        "turn": TurnInstruction(goal="g", technique="t").model_dump(),
        "fix_instruction": "删除确定性措辞，改为初步印象",
    })
    assert resp.status_code == 200
    assert resp.json()["reply"]
