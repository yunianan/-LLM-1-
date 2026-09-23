from __future__ import annotations

from typing import Any


def execute_safety_policy(passport: dict[str, Any]) -> dict[str, Any]:
    """把L1结论转换为可验证动作；不重新进行临床判断。"""

    path = passport.get("decision_path")
    controls = list(dict.fromkeys(passport.get("flags") or []))
    response = str(passport.get("instruction_to_next") or "")
    if path == "clinical_crisis":
        commands = ["pause_normal_pipeline", "do_not_call_l2", "show_safety_transition"]
        return {
            "route": "safety",
            "terminal": False,
            "consume_message": True,
            "resume_l2_next_turn": False,
            "user_response": response,
            "waiting_for": "safety_handoff",
            "commands": commands,
            "applied_controls": commands + controls,
            "clear_controls": [],
        }
    if path == "safety_pause":
        commands = ["pause_normal_pipeline", "do_not_call_l2", "show_safety_transition"]
        return {
            "route": "safety",
            "terminal": False,
            "consume_message": True,
            "resume_l2_next_turn": False,
            "user_response": response,
            "waiting_for": "professional_handoff",
            "commands": commands,
            "applied_controls": commands + controls,
            "clear_controls": [],
        }
    if path == "security_reject":
        commands = ["reject_security_request", "do_not_call_l2", "keep_session_active"]
        return {
            "route": "security",
            "terminal": False,
            "consume_message": True,
            "resume_l2_next_turn": False,
            "user_response": response,
            "waiting_for": "free_text",
            "commands": commands,
            "applied_controls": commands,
            "clear_controls": [],
        }
    if path == "safety_assessment":
        commands = ["continue_l1_safety_assessment", "do_not_call_l2"]
        return {
            "route": "l1",
            "terminal": False,
            "consume_message": True,
            "resume_l2_next_turn": bool(passport.get("resume_l2_next_turn")),
            "user_response": response,
            "waiting_for": str(passport.get("waiting_for") or "safety_answer"),
            "commands": commands,
            "applied_controls": commands + controls,
            "clear_controls": [],
        }
    commands = list(controls)
    return {
        "route": "l2",
        "terminal": False,
        "consume_message": False,
        "resume_l2_next_turn": False,
        "user_response": "",
        "waiting_for": None,
        "commands": commands,
        "applied_controls": commands,
        "clear_controls": [],
    }
