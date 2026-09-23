from typing import Any


CHECK_WEIGHTS = {
    "criteria_evidence": 0.25,
    "core_criteria": 0.20,
    "duration_and_impairment": 0.15,
    "exclusion_completeness": 0.20,
    "differential_diagnosis": 0.10,
    "evidence_consistency": 0.10,
}


class IndependentVerifier:
    """依据患者画像与逐条证据独立复核，不消费候选置信度。"""

    def verify(
        self,
        primary: dict[str, Any] | None,
        profile: dict[str, Any],
        initial_missing: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if primary is None:
            return {
                "score": 0.0,
                "routing": "reject",
                "checks": [],
                "fail_reasons": ["没有可供复核的候选诊断"],
                "missing_dimensions": list(initial_missing or []),
            }

        checks: list[dict[str, Any]] = []
        missing = list(initial_missing or [])
        fail_reasons: list[str] = []
        details = primary.get("dsm5_criteria_detail") or []
        met_details = [item for item in details if item.get("status") == "met"]

        bound_count = sum(bool(item.get("evidence_refs")) for item in met_details)
        evidence_score = bound_count / len(met_details) if met_details else 0.0
        self._add_check(
            checks, "criteria_evidence", evidence_score,
            "所有满足的 DSM 条目均有逐条证据" if evidence_score == 1.0
            else "存在缺少证据的 DSM 条目",
            [ref for item in met_details for ref in item.get("evidence_refs", [])],
        )

        met_codes = {item.get("code") for item in met_details}
        core_ok = len(met_codes) >= 5 and bool({"A1", "A2"} & met_codes)
        self._add_check(
            checks, "core_criteria", 1.0 if core_ok else 0.0,
            "MDD 数量门槛和核心条目满足" if core_ok else "MDD 数量门槛或核心条目不满足",
            sorted(code for code in met_codes if code),
        )
        if not core_ok:
            fail_reasons.append("MDD 核心诊断条件不满足")

        duration = profile.get("symptom_duration_days")
        impairment = profile.get("functional_impairment")
        duration_impairment_ok = duration is not None and impairment is not None
        if duration is None:
            missing.append(self._missing(
                "symptom_duration_days", "确认症状是否持续至少两周",
                "这些感受和变化大约持续多久了？",
            ))
        if impairment is None:
            missing.append(self._missing(
                "functional_impairment", "确认是否存在明显痛苦或功能损害",
                "这些情况对工作、学习、人际或日常生活造成了什么影响？",
            ))
        duration_impairment_pass = (
            duration_impairment_ok and int(duration) >= 14 and impairment is True
        )
        self._add_check(
            checks, "duration_and_impairment", 1.0 if duration_impairment_pass else 0.0,
            "持续时间与功能损害满足" if duration_impairment_pass else "持续时间或功能损害不满足/未知",
            [f"symptom_duration_days={duration}", f"functional_impairment={impairment}"],
        )

        mania = profile.get("mania_or_hypomania_history")
        attributable_cause = profile.get("substance_or_medical_cause")
        if mania is None:
            missing.append(self._missing(
                "mania_or_hypomania_history", "排除双相障碍",
                "过去是否有过持续数天精力异常旺盛、睡得很少却不困的时期？",
            ))
        if attributable_cause is None:
            missing.append(self._missing(
                "substance_or_medical_cause", "排除物质、药物或躯体疾病影响",
                "这些症状是否可能与药物、物质使用或身体疾病有关？",
            ))
        exclusions_known = mania is not None and attributable_cause is not None
        exclusions_pass = exclusions_known and mania is False and attributable_cause is False
        self._add_check(
            checks, "exclusion_completeness", 1.0 if exclusions_pass else 0.0,
            "关键排除条件完整且未命中" if exclusions_pass else "关键排除条件缺失或命中",
            [f"mania_or_hypomania_history={mania}", f"substance_or_medical_cause={attributable_cause}"],
        )
        if mania is True:
            fail_reasons.append("存在躁狂或轻躁狂史，当前 MDD 主诊断不能通过")
        if attributable_cause is True:
            fail_reasons.append("症状可能由物质、药物或躯体疾病导致")

        differentials = profile.get("differential_diagnoses_assessed")
        if differentials is None:
            missing.append(self._missing(
                "differential_diagnoses_assessed", "确认重要鉴别诊断已被评估",
                "请补充与双相障碍、适应障碍等相关的鉴别信息。",
            ))
        differential_pass = isinstance(differentials, list) and "Bipolar" in differentials
        self._add_check(
            checks, "differential_diagnosis", 1.0 if differential_pass else 0.0,
            "已评估双相障碍等重要鉴别诊断" if differential_pass else "重要鉴别诊断评估不足",
            list(differentials or []),
        )

        conflicts = profile.get("evidence_conflicts")
        if conflicts is None:
            missing.append(self._missing(
                "evidence_conflicts", "确认量表、患者叙述和观察是否一致",
                "请核对量表结果与患者陈述是否存在矛盾。",
            ))
        consistency_pass = isinstance(conflicts, list) and not conflicts
        self._add_check(
            checks, "evidence_consistency", 1.0 if consistency_pass else 0.0,
            "未发现证据冲突" if consistency_pass else "存在或尚未检查证据冲突",
            [str(item) for item in conflicts or []],
        )

        missing = self._dedupe_missing(missing)
        score = round(sum(check["score"] * CHECK_WEIGHTS[check["check_id"]] for check in checks), 4)
        if isinstance(conflicts, list) and conflicts:
            routing = "manual_review"
            fail_reasons.append("患者画像中存在相互冲突的证据")
        elif mania is True or attributable_cause is True:
            routing = "reject"
        elif missing:
            routing = "return_to_l2"
        elif evidence_score < 1.0 or not core_ok or not duration_impairment_pass:
            routing = "reject"
        elif score == 1.0:
            routing = "pass"
        elif score >= 0.5:
            routing = "conditional"
        else:
            routing = "reject"

        return {
            "score": score,
            "routing": routing,
            "checks": checks,
            "fail_reasons": fail_reasons,
            "missing_dimensions": missing,
        }

    @staticmethod
    def _add_check(checks, check_id, score, reason, evidence_refs):
        checks.append({
            "check_id": check_id,
            "status": "pass" if score == 1.0 else "fail",
            "score": score,
            "weight": CHECK_WEIGHTS[check_id],
            "reasons": [reason],
            "evidence_refs": evidence_refs,
        })

    @staticmethod
    def _missing(field: str, purpose: str, suggested_question: str) -> dict[str, str]:
        return {"field": field, "purpose": purpose, "suggested_question": suggested_question}

    @staticmethod
    def _dedupe_missing(items: list[dict[str, str]]) -> list[dict[str, str]]:
        result = []
        seen = set()
        for item in items:
            if item["field"] in seen:
                continue
            seen.add(item["field"])
            result.append(item)
        return result
