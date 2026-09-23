from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SUPPORTED_SCALES = {"PHQ-9", "GAD-7", "PCL-5", "Y-BOCS", "ISI", "MDQ"}
SUPPORTED_CONTROLS = {
    "avoid_trauma_probing",
    "monitor_suicidal_ideation",
    *(f"skip_scale:{scale}" for scale in SUPPORTED_SCALES),
}


class L2EntranceGuardError(RuntimeError):
    def __init__(self, code: str, violations: list[str]) -> None:
        super().__init__("；".join(violations))
        self.code = code
        self.violations = violations


class L2EntranceGuard:
    """只验证 L1 结构化结果，不重新分析患者自然语言。"""

    def __init__(self, *, max_age_seconds: int = 300) -> None:
        self.max_age_seconds = max_age_seconds

    def verify(self, passport: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        violations: list[str] = []
        if not isinstance(passport, dict):
            raise L2EntranceGuardError("L2_GUARD_SCHEMA", ["L1结果必须是对象"])

        if passport.get("layer_id") != "L1":
            violations.append("layer_id 必须为 L1")
        if passport.get("schema_version") != "1.0":
            violations.append("schema_version 必须为 1.0")
        bound_session = passport.get("session_id")
        if not isinstance(bound_session, str) or not bound_session:
            violations.append("缺少 L1 会话绑定")
        elif bound_session != session_id:
            violations.append("L1结果与当前会话不匹配")
        if not isinstance(passport.get("message_id"), str) or not passport.get("message_id"):
            violations.append("缺少 L1 消息绑定")

        timestamp = passport.get("timestamp")
        if timestamp is None:
            violations.append("缺少 L1 签发时间")
        else:
            try:
                issued_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                if issued_at.tzinfo is None:
                    raise ValueError("缺少时区")
                age = (datetime.now(timezone.utc) - issued_at).total_seconds()
                if age < -30 or age > self.max_age_seconds:
                    violations.append("L1结果已过期或时间异常")
            except (TypeError, ValueError):
                violations.append("timestamp 不是有效的带时区时间")

        risk_level = passport.get("risk_level")
        if isinstance(risk_level, bool) or not isinstance(risk_level, int) or not 0 <= risk_level <= 3:
            violations.append("risk_level 必须是 0-3 的整数")
        safety_flag = passport.get("safety_flag")
        if safety_flag not in {"pass", "warn", "block"}:
            violations.append("safety_flag 非法")
        crisis_flag = passport.get("crisis_flag")
        if not isinstance(crisis_flag, bool):
            violations.append("crisis_flag 必须是布尔值")
        decision_path = passport.get("decision_path")
        if decision_path != "normal":
            violations.append("L1决策路径不允许进入L2")

        labels = passport.get("structured_labels")
        if not isinstance(labels, dict):
            violations.append("缺少L1结构化安全状态")
            labels = {}
        handoff = labels.get("l2_handoff")
        if not isinstance(handoff, dict):
            violations.append("缺少L1到L2交接状态")
        elif handoff.get("ordinary_care_allowed") is not True:
            violations.append("L1尚未允许普通评估")
        asq = labels.get("asq")
        bssa = labels.get("bssa")
        if isinstance(asq, dict) and asq.get("status") in {
            "in_progress", "non_acute_positive", "acute_positive", "refused"
        }:
            completed_nonacute = (
                asq.get("status") in {"non_acute_positive", "refused"}
                and isinstance(bssa, dict)
                and bssa.get("status") == "completed"
            )
            if not completed_nonacute:
                violations.append("ASQ/BSSA安全流程尚未完成")
        if isinstance(bssa, dict) and bssa.get("status") == "in_progress":
            violations.append("BSSA安全评估仍在进行中")

        execution = passport.get("execution")
        guard = passport.get("guard")
        if not isinstance(execution, dict):
            violations.append("缺少 L1 策略执行结果")
            execution = {}
        if not isinstance(guard, dict):
            violations.append("缺少 L1 出口守卫回执")
            guard = {}

        blocked = risk_level == 3 or safety_flag == "block" or crisis_flag is True
        if blocked:
            violations.append("L1 已终止常规流程，禁止进入 L2")
        if execution.get("route") != "l2" or execution.get("terminal") is not False:
            violations.append("L1 未明确路由至 L2")
        if guard.get("status") != "pass" or guard.get("can_call_l2") is not True:
            violations.append("L1 出口守卫未允许调用 L2")

        applied = execution.get("applied_controls")
        if not isinstance(applied, list) or any(not isinstance(item, str) for item in applied):
            violations.append("applied_controls 必须是字符串列表")
            applied = []
        unsupported = sorted(set(applied) - SUPPORTED_CONTROLS)
        if unsupported:
            violations.append("L2不支持的控制命令：" + ",".join(unsupported))
        clear_controls = execution.get("clear_controls") or []
        if not isinstance(clear_controls, list) or any(
            not isinstance(item, str) for item in clear_controls
        ):
            violations.append("clear_controls 必须是字符串列表")
        else:
            unsupported_clear = sorted(set(clear_controls) - SUPPORTED_CONTROLS)
            if unsupported_clear:
                violations.append(
                    "L2不支持解除的控制命令：" + ",".join(unsupported_clear)
                )
        required = self.required_controls(passport)
        missing_controls = sorted(required - set(applied))
        if missing_controls:
            violations.append("L1命令缺少执行回执：" + ",".join(missing_controls))

        if risk_level == 2 and safety_flag != "warn":
            violations.append("risk_level=2 时 safety_flag 必须为 warn")
        if risk_level in {0, 1} and safety_flag != "pass":
            violations.append("risk_level=0/1 时 safety_flag 必须为 pass")

        if violations:
            raise L2EntranceGuardError("L2_GUARD_REJECTED", violations)
        return {
            "status": "pass",
            "route": "allow_l2",
            "verified_controls": sorted(required),
            "session_id": session_id,
        }

    @staticmethod
    def required_controls(passport: dict[str, Any]) -> set[str]:
        required: set[str] = set()
        flags = set(passport.get("flags") or [])
        if "avoid_trauma_probing" in flags:
            required.add("avoid_trauma_probing")
        if "monitor_suicidal_ideation" in flags:
            required.add("monitor_suicidal_ideation")
        labels = passport.get("structured_labels") or {}
        if isinstance(labels, dict):
            for scale in labels.get("scale_skip_list") or []:
                if isinstance(scale, str) and scale:
                    required.add(f"skip_scale:{scale}")
        return required


def merge_persistent_controls(
    current: list[str], passport: dict[str, Any]
) -> list[str]:
    """安全控制持续生效，只有 L1 的明确 clear_controls 才能解除。"""

    execution = passport.get("execution") or {}
    applied = set(execution.get("applied_controls") or [])
    cleared = set(execution.get("clear_controls") or [])
    return sorted((set(current) | applied) - cleared)
