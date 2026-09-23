from __future__ import annotations

import uuid
from typing import Any

from .risk_classifier import llm_classify
from .service import L1SafetyService


_DEFAULT_SERVICE = L1SafetyService()


def reset_session(session_id: str) -> None:
    _DEFAULT_SERVICE.reset_session(session_id)


def start_baseline_asq(
    *, session_id: str, message_id: str | None = None
) -> dict[str, Any] | None:
    return _DEFAULT_SERVICE.start_baseline_asq(
        session_id=session_id,
        message_id=message_id,
    )


def safety_detect(
    text: str,
    *,
    session_id: str | None = None,
    message_id: str | None = None,
    assessment_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sid = session_id or f"standalone-{uuid.uuid4().hex}"
    mid = message_id or uuid.uuid4().hex
    return _DEFAULT_SERVICE.handle_message(
        text,
        session_id=sid,
        message_id=mid,
        assessment_context=assessment_context,
        classifier=llm_classify,
    )
