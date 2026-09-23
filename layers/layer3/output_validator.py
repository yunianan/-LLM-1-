from typing import Any

from shared.report_integrity import REQUIRED_L3_CHECKS, l3_report_fingerprint
from .instructions.instruction_generator import validate_session_instructions


MIN_CANDIDATE_CONFIDENCE = 0.5

PRIMARY_THERAPY_BY_DIAGNOSIS = {
    "MDD": {"CBT", "CBT+BA"},
    "GAD": {"CBT"},
    "PTSD": {"CBT-TF"},
    "OCD": {"CBT"},
    "BIPOLAR": {"CBT+IPSRT"},
    "INSOMNIA": {"CBT-I"},
    "PANICDISORDER": {"CBT"},
    "SOCIALANXIETYDISORDER": {"CBT"},
    "ADJUSTMENTDISORDER": {"CBT/SFBT"},
    "SOMATICSYMPTOMDISORDER": {"CBT"},
}

DIAGNOSIS_ALIASES = {
    "MDD": "MDD",
    "MAJORDEPRESSIVEDISORDER": "MDD",
    "GAD": "GAD",
    "GENERALIZEDANXIETYDISORDER": "GAD",
    "PTSD": "PTSD",
    "POSTTRAUMATICSTRESSDISORDER": "PTSD",
    "OCD": "OCD",
    "OBSESSIVECOMPULSIVEDISORDER": "OCD",
    "BIPOLAR": "BIPOLAR",
    "BIPOLARDISORDER": "BIPOLAR",
    "BIPOLARI": "BIPOLAR",
    "BIPOLARII": "BIPOLAR",
    "INSOMNIA": "INSOMNIA",
    "INSOMNIADISORDER": "INSOMNIA",
}

class L3OutputValidator:
    """L3 对自身诊断输出进行确定性自检，并生成供 L4 验证的凭证。"""

    def validate(self, report: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, str]] = []
        diagnosis = report.get("diagnosis") or {}
        primary = diagnosis.get("primary")

        if primary is None:
            for rule_id in sorted(REQUIRED_L3_CHECKS):
                self._check(checks, rule_id, False, "缺少主诊断，无法执行该检查")
            return self._receipt(report, checks)

        try:
            validate_session_instructions(report)
            self._check(checks, "V3-TREATMENT-INSTRUCTIONS", True, "会话指令与诊断、疗法一致")
        except (ValueError, KeyError, TypeError) as exc:
            self._check(checks, "V3-TREATMENT-INSTRUCTIONS", False, str(exc))

        candidate_confidence = (
            report["candidate_confidence"]
            if "candidate_confidence" in report
            else primary.get("confidence", 0.0)
        )
        self._check(
            checks,
            "V3-CANDIDATE-CONFIDENCE",
            candidate_confidence >= MIN_CANDIDATE_CONFIDENCE,
            f"候选诊断支持度={candidate_confidence}，最低门槛={MIN_CANDIDATE_CONFIDENCE}",
        )

        primary_code = self._canonical_diagnosis(primary.get("disorder", ""))
        ruled_out_codes = {
            self._canonical_diagnosis(item.get("disorder", ""))
            for item in diagnosis.get("ruled_out") or []
        }
        self._check(
            checks,
            "V3-PRIMARY-NOT-RULED-OUT",
            primary_code not in ruled_out_codes,
            f"主诊断={primary.get('disorder', '')}，排除诊断={sorted(ruled_out_codes)}",
        )

        detail_by_code = {
            item.get("code"): item
            for item in primary.get("dsm5_criteria_detail") or []
        }
        missing_evidence = [
            code
            for code in primary.get("dsm5_criteria_met") or []
            if code not in detail_by_code
            or detail_by_code[code].get("status") != "met"
            or not detail_by_code[code].get("evidence_refs")
        ]
        self._check(
            checks,
            "V3-CRITERIA-EVIDENCE",
            not missing_evidence,
            f"缺少逐条证据的DSM条目={missing_evidence}",
        )

        therapy = (report.get("recommended_therapy") or {}).get("primary", "")
        allowed = PRIMARY_THERAPY_BY_DIAGNOSIS.get(primary_code)
        therapy_ok = allowed is not None and therapy in allowed
        self._check(
            checks,
            "V3-THERAPY-MATCH",
            therapy_ok,
            f"主诊断={primary.get('disorder', '')}，疗法={therapy or '<empty>'}，允许值={sorted(allowed or [])}",
        )

        recommended = report.get("recommended_therapy") or {}
        plugins = recommended.get("addon") or []
        triggers = recommended.get("plugin_triggers") or []
        trigger_plugins = [item.get("plugin") for item in triggers]
        plugin_routing_ok = (
            len(plugins) <= 2
            and len(plugins) == len(set(plugins))
            and plugins == trigger_plugins
        )
        self._check(
            checks,
            "V3-PLUGIN-ROUTING",
            plugin_routing_ok,
            f"插件={plugins}，触发记录={trigger_plugins}，最多允许2个且必须逐项携带触发依据",
        )

        adjustments = recommended.get("strategy_adjustments") or {}
        adjustment_ok = bool(adjustments.get("severity_band")) and isinstance(
            adjustments.get("instructions"), list
        )
        self._check(
            checks,
            "V3-THERAPY-ADJUSTMENT",
            adjustment_ok,
            "Step 3必须输出严重程度分组和可供L4执行的微调指令",
        )

        exposure_blocked = adjustments.get("exposure_policy") == "prohibited_until_stabilized"
        exposure_plugins = {"EMDR", "ERP"} & set(plugins)
        addiction_first = adjustments.get("care_priority") == "addiction_treatment_first"
        contraindications = recommended.get("contraindications") or []
        requires_psychiatry = any("精神病性症状" in item for item in contraindications)
        requires_addiction_first = any("先处理成瘾" in item for item in contraindications)
        requires_stabilization = any("先做稳定化" in item for item in contraindications)
        care_priorities = set(adjustments.get("care_priorities") or [])
        if adjustments.get("care_priority"):
            care_priorities.add(adjustments["care_priority"])
        contraindication_ok = (
            not (exposure_blocked and exposure_plugins)
            and not (addiction_first and plugins)
            and (
                not requires_psychiatry
                or "psychiatric_referral" in care_priorities
            )
            and (
                not requires_addiction_first
                or "addiction_treatment_first" in care_priorities
            )
            and (
                not requires_stabilization
                or "stabilization_first" in care_priorities
            )
        )
        self._check(
            checks,
            "V3-CONTRAINDICATION-OVERRIDE",
            contraindication_ok,
            f"暴露禁用={exposure_blocked}，暴露插件={sorted(exposure_plugins)}，成瘾优先={addiction_first}",
        )

        return self._receipt(report, checks)

    @staticmethod
    def _check(checks, rule_id, passed, detail):
        checks.append({
            "rule_id": rule_id,
            "status": "pass" if passed else "fail",
            "detail": detail,
        })

    @staticmethod
    def _receipt(report, checks):
        passed = all(item["status"] == "pass" for item in checks)
        return {
            "passed": passed,
            "route": "allow_l4" if passed else "reject",
            "checks": checks,
            "payload_hash": l3_report_fingerprint(report),
            "policy_version": "L3_OUTPUT_V1",
        }

    @staticmethod
    def _canonical_diagnosis(value: str) -> str:
        normalized = "".join(ch for ch in value.upper() if ch.isalnum())
        return DIAGNOSIS_ALIASES.get(normalized, normalized)
