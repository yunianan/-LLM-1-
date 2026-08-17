import pytest

from l4.llm.base import Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.history import ConversationTurn, SessionData, SessionStore
from l4.session.state_machine import SUBSTAGE_ORDER, SubstageMachine


def test_substage_order():
    assert SUBSTAGE_ORDER == ("L4-L1", "L4-L2", "L4-L3", "L4-L4")


def test_default_forward_progression():
    m = SubstageMachine()
    assert m.resolve() == "L4-L1"
    m.advance(stage_complete=True)
    assert m.current == "L4-L2"
    m.advance(stage_complete=True)
    assert m.current == "L4-L3"
    m.advance(stage_complete=True)
    assert m.current == "L4-L4"
    m.advance(stage_complete=True)
    assert m.current == "L4-L4"  # 终态不越界
    assert "terminal_reached" in m.flags


def test_force_substage_overrides():
    m = SubstageMachine(current="L4-L1")
    assert m.resolve(force="L4-L3") == "L4-L3"
    m.advance(force="L4-L3")
    assert m.current == "L4-L3"


def test_regression_override_flagged():
    m = SubstageMachine(current="L4-L3")
    m.advance(force="L4-L1")
    assert m.current == "L4-L1"
    assert "stage_regression_override" in m.flags


def test_not_complete_no_advance():
    m = SubstageMachine(current="L4-L2")
    m.advance(stage_complete=False)
    assert m.current == "L4-L2"


def test_flags_deduplicated():
    m = SubstageMachine(current="L4-L2")
    m.advance(force="L4-L1")
    m.advance(force="L4-L1")
    assert m.flags.count("stage_regression_override") == 1


def _make_session_data(session_id="s1") -> SessionData:
    return SessionData(
        session_id=session_id,
        machine=SubstageMachine(),
        role=RoleInstruction(identity="CBT治疗师", style="温暖"),
        anchor=None,
        boundary=BoundaryInstruction(),
        safety=SafetyPassport(),
        therapy_options=["CBT", "CBT-I"],
    )


def test_session_store_lifecycle():
    store = SessionStore()
    assert not store.exists("s1")
    data = _make_session_data()
    store.create(data)
    assert store.exists("s1")
    assert store.get("s1").session_id == "s1"
    with pytest.raises(KeyError):
        store.get("nope")


def test_session_get_messages_and_history():
    data = _make_session_data()
    data.history.append(ConversationTurn(role="user", content="你好"))
    data.history.append(ConversationTurn(role="assistant", content="你好，今天想聊点什么？"))
    msgs = data.get_messages()
    assert msgs == [
        Message(role="user", content="你好"),
        Message(role="assistant", content="你好，今天想聊点什么？"),
    ]
