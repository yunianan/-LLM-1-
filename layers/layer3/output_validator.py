from typing import Any

from shared.report_integrity import REQUIRED_L3_CHECKS, l3_report_fingerprint


MIN_CANDIDATE_CONFIDENCE = 0.5

PRIMARY_THERAPY_BY_DIAGNOSIS = {
    "MDD": {"CBT", "CBT+BA"},
    "GAD": {"CBT"},
    "PTSD": {"CBT-TF"},
    "OCD": {"CBT"},
    "BIPOLAR": {"CBT+IPSRT"},
    "INSOMNIA": {"CBT-I"},
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
