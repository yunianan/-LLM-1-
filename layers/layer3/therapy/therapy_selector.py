from __future__ import annotations

from typing import Any, Iterable


from shared.disorder_catalog import treatment_key

BASE_THERAPY_BY_DIAGNOSIS = {
    "MDD": "CBT+BA",
    "GAD": "CBT",
    "PTSD": "CBT-TF",
    "OCD": "CBT",
    "Insomnia": "CBT-I",
    "AdjustmentDisorder": "CBT/SFBT",
    "Bipolar": "CBT+IPSRT",
}

PLUGIN_PRIORITY = ["IPSRT", "EMDR", "ERP", "CBT-I", "MBCT", "IPT", "ACT", "SFBT"]
PLUGIN_PRIORITY_LABEL = {
    "IPSRT": "priority_0",
    "EMDR": "high",
    "ERP": "high",
    "CBT-I": "high",
    "MBCT": "regular",
    "IPT": "regular",
    "ACT": "regular",
    "SFBT": "situational",
}
EXPOSURE_PLUGINS = {"EMDR", "ERP"}


class TherapySelector:
    """按固定规则执行基础疗法、插件匹配和患者特征微调。"""

    def select_basic(self, diagnosis: str) -> str:
        try:
            return BASE_THERAPY_BY_DIAGNOSIS[treatment_key(diagnosis)]
        except KeyError as exc:
            raise ValueError(f"不支持的主诊断：{diagnosis}") from exc

    def select_plugin_triggers(
        self,
        diagnosis: str,
        profile: dict[str, Any],
        *,
        confirmed_diagnoses: Iterable[str] | None = None,
        verifier_decision: str = "confirmed",
    ) -> list[dict[str, Any]]:
        confirmed = self._confirmed_set(
            diagnosis,
            confirmed_diagnoses,
            verifier_decision,
        )

        triggers: list[dict[str, Any]] = []

        def add(plugin: str, conditions: list[str]) -> None:
            triggers.append({
                "plugin": plugin,
                "priority": PLUGIN_PRIORITY_LABEL[plugin],
                "conditions": conditions,
            })

        if (
            treatment_key(diagnosis) == "Bipolar"
            and self._score(profile, "MDQ") >= 7
            and bool(profile.get("rhythm_disturbance"))
        ):
            add("IPSRT", [
                f"MDQ={self._score(profile, 'MDQ'):g}≥7",
                "rhythm_disturbance=true",
                "primary=Bipolar",
            ])

        if (
            "PTSD" in confirmed
            and self._score(profile, "PCL5") >= 33
            and bool(profile.get("exposure_avoidant"))
        ):
            add("EMDR", [
                f"PCL5={self._score(profile, 'PCL5'):g}≥33",
                "exposure_avoidant=true",
                "PTSD=confirmed",
            ])

        if (
            "OCD" in confirmed
            and self._score(profile, "Y_BOCS") >= 16
            and bool(profile.get("ritual_behaviors"))
        ):
            add("ERP", [
                f"Y_BOCS={self._score(profile, 'Y_BOCS'):g}≥16",
                "ritual_behaviors=true",
                "OCD=confirmed",
            ])

        if (
            bool(profile.get("insomnia_chief_complaint"))
            and self._score(profile, "ISI") >= 15
            and self._score(profile, "chronic_months") >= 3
        ):
            add("CBT-I", [
                f"ISI={self._score(profile, 'ISI'):g}≥15",
                f"chronic_months={self._score(profile, 'chronic_months'):g}≥3",
                "insomnia_chief_complaint=true",
            ])

        if "MDD" in confirmed and (
            self._score(profile, "relapse_count") >= 3
            or self._score(profile, "RRS") >= 44
        ):
            add("MBCT", [
                f"relapse_count={self._score(profile, 'relapse_count'):g}≥3 or "
                f"RRS={self._score(profile, 'RRS'):g}≥44",
                "MDD=confirmed",
            ])

        if (
            self._score(profile, "interpersonal_conflicts") >= 2
            and bool(profile.get("interpersonal_impairment"))
        ):
            add("IPT", [
                f"interpersonal_conflicts={self._score(profile, 'interpersonal_conflicts'):g}≥2",
                "interpersonal_impairment=true",
            ])

        if (
            self._score(profile, "AAQ_II") >= 24
            or self._score(profile, "PHQ15") >= 10
        ):
            add("ACT", [
                f"AAQ_II={self._score(profile, 'AAQ_II'):g}≥24 or "
                f"PHQ15={self._score(profile, 'PHQ15'):g}≥10"
            ])

        if (
            "AdjustmentDisorder" in confirmed
            and self._score(profile, "ADNM") >= 47.5
            and bool(profile.get("single_focus"))
        ):
            add("SFBT", [
                f"ADNM={self._score(profile, 'ADNM'):g}≥47.5",
                "single_focus=true",
                "AdjustmentDisorder=confirmed",
            ])

        triggers.sort(key=lambda item: PLUGIN_PRIORITY.index(item["plugin"]))
        return triggers

    def select_plugins(
        self,
        diagnosis: str,
        profile: dict[str, Any],
        *,
        confirmed_diagnoses: Iterable[str] | None = None,
        verifier_decision: str = "confirmed",
    ) -> list[str]:
        active, _ = self._route_plugin_triggers(
            diagnosis,
            profile,
            confirmed_diagnoses=confirmed_diagnoses,
            verifier_decision=verifier_decision,
        )
        return [item["plugin"] for item in active]

    def contraindications(
        self,
        diagnosis: str,
        profile: dict[str, Any],
        *,
        confirmed_diagnoses: Iterable[str] | None = None,
        verifier_decision: str = "confirmed",
    ) -> list[str]:
        contraindications: list[str] = []
        confirmed = self._confirmed_set(
            diagnosis,
            confirmed_diagnoses,
            verifier_decision,
        )
        if treatment_key(diagnosis) == "Bipolar":
            contraindications.append("双相障碍建议同时就医")
        if "PTSD" not in confirmed and self._score(profile, "PCL5") >= 33:
            contraindications.append("不推荐EMDR（无PTSD诊断证据）")
        if profile.get("psychotic_symptoms"):
            contraindications.append("合并精神病性症状→建议精神科就医")
        if profile.get("substance_active"):
            contraindications.append("物质依赖活跃期→先处理成瘾")
        if profile.get("severe_dissociation"):
            contraindications.append("严重解离→先做稳定化，不做暴露")
        return contraindications

    def adjust_plan(
        self,
        profile: dict[str, Any],
        *,
        suppressed_plugins: list[dict[str, str]],
    ) -> dict[str, Any]:
        phq9 = self._optional_score(profile, "PHQ9")
        gad7 = self._optional_score(profile, "GAD7")

        if (phq9 is not None and phq9 >= 20) or (gad7 is not None and gad7 >= 15):
            severity_band = "severe"
            technical_density = "reduced"
            supportive_response = "increased"
            safety_monitoring = "enhanced"
            intervention_focus = ["supportive_response", "paced_techniques"]
            instructions = ["降低每轮技术密度", "增加支持性回应比例", "强化安全监控频率"]
        elif (phq9 is not None and phq9 >= 10) or (gad7 is not None and gad7 >= 8):
            severity_band = "moderate"
            technical_density = "standard"
            supportive_response = "standard"
            safety_monitoring = "standard"
            intervention_focus = ["standard_cbt_seven_steps", "activated_plugins"]
            instructions = ["使用标准CBT七步流程并整合已激活插件"]
        elif (
            phq9 is not None
            and gad7 is not None
            and phq9 < 10
            and gad7 < 8
        ):
            severity_band = "mild"
            technical_density = "light"
            supportive_response = "standard"
            safety_monitoring = "standard"
            intervention_focus = ["behavioral_activation", "self_help_materials"]
            instructions = ["聚焦行为激活", "提供自助材料", "减少结构化干预"]
        else:
            severity_band = "undetermined"
            technical_density = "standard"
            supportive_response = "standard"
            safety_monitoring = "standard"
            intervention_focus = []
            instructions = ["严重程度信息不足时保持标准节奏，不推断轻重程度"]

        motivation = profile.get("motivation") or {}
        importance = self._optional_score(motivation, "importance")
        confidence = self._optional_score(motivation, "confidence")
        motivation_total = (
            importance + confidence
            if importance is not None and confidence is not None
            else None
        )
        if motivation_total is not None and motivation_total >= 14:
            motivation_band = "high"
            challenge_interventions = "early_allowed"
            instructions.append("可较早引入暴露练习或行为实验等挑战性干预")
        elif motivation_total is not None and motivation_total < 8:
            motivation_band = "low"
            challenge_interventions = "deferred"
            instructions.extend(["优先使用MI唤出技术", "暂缓行动导向干预，避免推得过快"])
        elif motivation_total is not None:
            motivation_band = "intermediate"
            challenge_interventions = "standard_timing"
        else:
            motivation_band = "unknown"
            challenge_interventions = "standard_timing"

        care_priority = "psychotherapy"
        care_priorities: list[str] = []
        exposure_policy = "standard"
        if profile.get("psychotic_symptoms"):
            care_priority = "psychiatric_referral"
            care_priorities.append("psychiatric_referral")
            challenge_interventions = "deferred"
            instructions = [
                item for item in instructions
                if item != "可较早引入暴露练习或行为实验等挑战性干预"
            ]
            instructions.append("合并精神病性症状，建议精神科就医")
        if profile.get("substance_active"):
            care_priority = "addiction_treatment_first"
            care_priorities.append("addiction_treatment_first")
            challenge_interventions = "deferred"
            instructions = [
                item for item in instructions
                if item != "可较早引入暴露练习或行为实验等挑战性干预"
            ]
            instructions.append("物质依赖活跃期先处理成瘾，暂缓行动导向心理干预")
        if profile.get("severe_dissociation"):
            care_priority = "stabilization_first"
            care_priorities.append("stabilization_first")
            exposure_policy = "prohibited_until_stabilized"
            challenge_interventions = "deferred"
            instructions = [
                item for item in instructions
                if item != "可较早引入暴露练习或行为实验等挑战性干预"
            ]
            instructions.append("严重解离时先做稳定化，不进行暴露")
        if not care_priorities:
            care_priorities.append("psychotherapy")

        return {
            "severity_band": severity_band,
            "motivation_band": motivation_band,
            "motivation_total": motivation_total,
            "technical_density": technical_density,
            "supportive_response": supportive_response,
            "safety_monitoring": safety_monitoring,
            "intervention_focus": intervention_focus,
            "challenge_interventions": challenge_interventions,
            "exposure_policy": exposure_policy,
            "care_priority": care_priority,
            "care_priorities": care_priorities,
            "instructions": list(dict.fromkeys(instructions)),
            "suppressed_plugins": suppressed_plugins,
        }

    def run(
        self,
        diagnosis: str,
        profile: dict[str, Any],
        *,
        confirmed_diagnoses: Iterable[str] | None = None,
        verifier_decision: str = "confirmed",
    ) -> dict[str, Any]:
        basic = self.select_basic(diagnosis)
        active_triggers, suppressed = self._route_plugin_triggers(
            diagnosis,
            profile,
            confirmed_diagnoses=confirmed_diagnoses,
            verifier_decision=verifier_decision,
        )

        return {
            "basic_therapy": basic,
            "plugins": [item["plugin"] for item in active_triggers],
            "plugin_triggers": active_triggers,
            "contraindications": self.contraindications(
                diagnosis,
                profile,
                confirmed_diagnoses=confirmed_diagnoses,
                verifier_decision=verifier_decision,
            ),
            "strategy_adjustments": self.adjust_plan(
                profile,
                suppressed_plugins=suppressed,
            ),
        }

    def _route_plugin_triggers(
        self,
        diagnosis: str,
        profile: dict[str, Any],
        *,
        confirmed_diagnoses: Iterable[str] | None,
        verifier_decision: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        all_triggers = self.select_plugin_triggers(
            diagnosis,
            profile,
            confirmed_diagnoses=confirmed_diagnoses,
            verifier_decision=verifier_decision,
        )
        active_triggers: list[dict[str, Any]] = []
        suppressed: list[dict[str, str]] = []
        for trigger in all_triggers:
            reason = self._plugin_block_reason(trigger["plugin"], profile)
            if reason:
                suppressed.append({"plugin": trigger["plugin"], "reason": reason})
            else:
                active_triggers.append(trigger)
        return active_triggers[:2], suppressed

    @staticmethod
    def _plugin_block_reason(plugin: str, profile: dict[str, Any]) -> str:
        if profile.get("substance_active"):
            return "物质依赖活跃期先处理成瘾"
        if profile.get("severe_dissociation") and plugin in EXPOSURE_PLUGINS:
            return "严重解离时先稳定化，不进行暴露"
        return ""

    @staticmethod
    def _confirmed_set(
        diagnosis: str,
        confirmed_diagnoses: Iterable[str] | None,
        verifier_decision: str,
    ) -> set[str]:
        if verifier_decision != "confirmed":
            return set()
        confirmed = {treatment_key(item) for item in confirmed_diagnoses or ()}
        if not confirmed:
            confirmed.add(treatment_key(diagnosis))
        return confirmed

    @staticmethod
    def _optional_score(source: dict[str, Any], key: str) -> float | None:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    @classmethod
    def _score(cls, source: dict[str, Any], key: str) -> float:
        value = cls._optional_score(source, key)
        return value if value is not None else 0.0
