from typing import Any


class L1GuardError(RuntimeError):
    pass


class L1ExitGuard:
    """验证 L1 的风险判断已被策略执行器正确落实。"""

    def verify(self, passport: dict[str, Any]) -> dict[str, Any]:
        violations: list[str] = []
        execution = passport.get("execution") or {}
        flag = passport.get("safety_flag")
        risk_level = passport.get("risk_level")
        crisis_flag = passport.get("crisis_flag")
        applied = set(execution.get("applied_controls") or [])
        flags = set(passport.get("flags") or [])

        if flag not in {"pass", "warn", "block"}:
            violations.append("safety_flag 非法")
        if not isinstance(risk_level, int) or not 0 <= risk_level <= 3:
            violations.append("risk_level 必须是 0-3 的整数")
        if risk_level == 3 and crisis_flag is not True:
            violations.append("risk_level=3 时 crisis_flag 必须为 true")
        if crisis_flag is True and flag != "block":
            violations.append("crisis_flag=true 时 safety_flag 必须为 block")

        blocked = flag == "block" or crisis_flag is True
        if crisis_flag is True:
            if execution.get("route") != "crisis":
                violations.append("block 决定必须路由至 crisis")
            if execution.get("terminal") is not True:
                violations.append("block 决定必须终止常规流程")
            if execution.get("crisis_response_generated") is not True:
                violations.append("危机路由必须生成安全响应")
            if not str(execution.get("user_response") or "").strip():
                violations.append("危机路由必须包含非空用户响应")
            for command in ("stop_normal_pipeline", "do_not_call_l2", "show_crisis_response"):
                if command not in applied:
                    violations.append(f"危机命令未执行：{command}")
        elif blocked:
            violations.append("block 决定缺少可验证的危机或LLM降级原因")
        else:
            if execution.get("route") != "l2":
                violations.append("非阻断决定应路由至 L2")
            if execution.get("terminal") is not False:
                violations.append("非阻断决定不应终止常规流程")

        required_controls = set()
        if "avoid_trauma_probing" in flags:
            required_controls.add("avoid_trauma_probing")
        if "monitor_suicidal_ideation" in flags:
            required_controls.add("monitor_suicidal_ideation")
        labels = passport.get("structured_labels") or {}
        required_controls.update(
            f"skip_scale:{scale}" for scale in labels.get("scale_skip_list") or []
        )
        for control in sorted(required_controls - applied):
            violations.append(f"L1 控制命令未执行：{control}")

        return {
            "status": "pass" if not violations else "fail",
            "can_call_l2": not blocked and not violations,
            "violations": violations,
            "verified_controls": sorted(required_controls & applied),
        }

    def assert_valid(self, passport: dict[str, Any]) -> dict[str, Any]:
        result = self.verify(passport)
        if result["status"] != "pass":
            raise L1GuardError("；".join(result["violations"]))
        return result
