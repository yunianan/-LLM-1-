from __future__ import annotations

from typing import Any


class L1GuardError(RuntimeError):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("；".join(violations))
        self.violations = violations


class L1ExitGuard:
    """验证L1决策已被执行，并严格控制L2入口。"""

    def verify(self, passport: dict[str, Any]) -> dict[str, Any]:
        violations: list[str] = []
        path = passport.get("decision_path")
        execution = passport.get("execution")
        if path not in {
            "normal", "safety_assessment", "safety_pause",
            "clinical_crisis", "security_reject",
        }:
            violations.append("decision_path非法")
        if not isinstance(execution, dict):
            violations.append("缺少策略执行结果")
            execution = {}

        expected_routes = {
            "normal": "l2",
            "safety_assessment": "l1",
            "safety_pause": "safety",
            "clinical_crisis": "safety",
            "security_reject": "security",
        }
        if path in expected_routes and execution.get("route") != expected_routes[path]:
            violations.append("策略执行路由与L1结论不一致")
        can_call_l2 = path == "normal" and execution.get("route") == "l2"
        if path != "normal" and execution.get("consume_message") is not True:
            violations.append("L1安全消息必须由L1消费，禁止传递给L2")
        if path in {"safety_pause", "clinical_crisis"}:
            commands = set(execution.get("commands") or [])
            if not {"pause_normal_pipeline", "do_not_call_l2"}.issubset(commands):
                violations.append("安全处置没有实际暂停普通疗程")
            if not str(execution.get("user_response") or "").strip():
                violations.append("安全处置缺少患者可见支持与行动信息")
        if path == "security_reject" and not str(execution.get("user_response") or "").strip():
            violations.append("越狱拒绝缺少患者可见回应")
        if path == "safety_assessment" and not str(execution.get("user_response") or "").strip():
            violations.append("安全评估缺少下一项患者问题")
        if execution.get("terminal") is not False:
            violations.append("L1基础流程不得直接关闭患者界面")

        return {
            "status": "fail" if violations else "pass",
            "can_call_l2": can_call_l2 and not violations,
            "violations": violations,
            "verified_controls": list(execution.get("applied_controls") or []),
        }

    def assert_valid(self, passport: dict[str, Any]) -> dict[str, Any]:
        result = self.verify(passport)
        if result["status"] != "pass":
            raise L1GuardError(result["violations"])
        return result
