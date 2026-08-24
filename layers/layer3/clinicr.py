from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

DISORDER_SEVERITY_SCALE = {
    "MDD": "PHQ9",
    "GAD": "GAD7",
    "PTSD": "PCL5",
    "Insomnia": "ISI",
    "Bipolar": "MDQ",
    "OCD": "Y_BOCS",
}

DSM5_CRITERIA_RULES = {
    "MDD": [
        ("A1", "心境抑郁", lambda p: p.get("PHQ9", 0) >= 10, "PHQ-9总分>=10"),
        ("A2", "兴趣减退", lambda p: p.get("PHQ9", 0) >= 10, "PHQ-9总分>=10"),
        ("A3", "食欲/体重改变", lambda p: p.get("PHQ9", 0) >= 15, "PHQ-9总分>=15"),
        ("A4", "睡眠障碍", lambda p: p.get("ISI", 0) >= 8, "ISI>=8"),
        ("A5", "精神运动性改变", lambda p: p.get("PHQ9", 0) >= 20, "PHQ-9总分>=20"),
        ("A6", "疲劳/精力减退", lambda p: p.get("PHQ9", 0) >= 15, "PHQ-9总分>=15"),
        ("A7", "无价值感/过度自责", lambda p: p.get("PHQ9", 0) >= 10, "PHQ-9总分>=10"),
        ("A8", "注意力减退", lambda p: p.get("PHQ9", 0) >= 20, "PHQ-9总分>=20"),
        ("A9", "自杀意念", lambda p: p.get("PHQ9", 0) >= 20, "PHQ-9总分>=20"),
    ],
    "GAD": [
        ("C1", "过度焦虑/担忧", lambda p: p.get("GAD7", 0) >= 8, "GAD-7总分>=8"),
        ("C2", "难以控制担忧", lambda p: p.get("GAD7", 0) >= 8, "GAD-7总分>=8"),
        ("C3", "坐立不安", lambda p: p.get("GAD7", 0) >= 10, "GAD-7总分>=10"),
        ("C4", "易疲劳", lambda p: p.get("GAD7", 0) >= 10, "GAD-7总分>=10"),
        ("C5", "注意力难以集中", lambda p: p.get("GAD7", 0) >= 10, "GAD-7总分>=10"),
        ("C6", "易怒", lambda p: p.get("GAD7", 0) >= 10, "GAD-7总分>=10"),
        ("C7", "肌肉紧张", lambda p: p.get("GAD7", 0) >= 10, "GAD-7总分>=10"),
        ("C8", "睡眠障碍", lambda p: p.get("ISI", 0) >= 8, "ISI>=8"),
    ],
    "PTSD": [
        ("A1", "创伤暴露", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
        ("B1", "闯入性回忆", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
        ("B2", "噩梦", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
        ("C1", "回避行为", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
        ("D1", "负性认知/情绪", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
        ("E1", "警觉性增高", lambda p: p.get("PCL5", 0) >= 31, "PCL-5总分>=31"),
    ],
    "Insomnia": [
        ("A1", "入睡困难", lambda p: p.get("ISI", 0) >= 8, "ISI>=8"),
        ("A2", "睡眠维持困难", lambda p: p.get("ISI", 0) >= 8, "ISI>=8"),
        ("A3", "早醒", lambda p: p.get("ISI", 0) >= 8, "ISI>=8"),
        ("A4", "日间功能受损", lambda p: p.get("ISI", 0) >= 15, "ISI>=15"),
    ],
    "Bipolar": [
        ("A1", "躁狂/轻躁狂发作史", lambda p: p.get("MDQ", 0) >= 7, "MDQ>=7"),
        ("B1", "活动增多/精力旺盛", lambda p: p.get("MDQ", 0) >= 7, "MDQ>=7"),
    ],
}


class CLINICR:
    def __init__(self, rules_path: str = "diagnosis_rules.json"):
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self.exclusion_rules = self.rules.get("exclusion_rules", {})
        self.severity_thresholds = self.rules.get("severity_thresholds", {})

    def _load_rules(self) -> Dict:
        return self._default_rules()

    def _default_rules(self) -> Dict:
        return {
            "rules": [
                {
                    "diagnosis": "MDD",
                    "label": "Major Depressive Disorder",
                    "conditions": {"PHQ9_min": 10},
                    "base_confidence": 0.75,
                    "evidence_template": "PHQ-9={PHQ9}, MDD criteria met"
                },
                {
                    "diagnosis": "GAD",
                    "label": "Generalized Anxiety Disorder",
                    "conditions": {"GAD7_min": 8},
                    "base_confidence": 0.45,
                    "evidence_template": "GAD-7={GAD7}, GAD criteria met"
                },
                {
                    "diagnosis": "PTSD",
                    "label": "Post-Traumatic Stress Disorder",
                    "conditions": {"PCL5_min": 31},
                    "base_confidence": 0.50,
                    "evidence_template": "PCL-5={PCL5}, PTSD criteria met"
                },
                {
                    "diagnosis": "Insomnia",
                    "label": "Insomnia Disorder",
                    "conditions": {"ISI_min": 15},
                    "base_confidence": 0.40,
                    "evidence_template": "ISI={ISI}, Insomnia criteria met"
                },
                {
                    "diagnosis": "Bipolar",
                    "label": "Bipolar Disorder",
                    "conditions": {"MDQ_min": 7},
                    "base_confidence": 0.55,
                    "evidence_template": "MDQ={MDQ}, Bipolar criteria met"
                }
            ],
            "exclusion_rules": {
                "MDD": {"exclude_if": {"MDQ": ">= 7"}},
                "GAD": {"exclude_if": {}},
                "PTSD": {"exclude_if": {}},
                "Insomnia": {"exclude_if": {}},
                "Bipolar": {"exclude_if": {}}
            },
            "severity_thresholds": {
                "PHQ9": {
                    "normal": {"min": 0, "max": 4},
                    "mild": {"min": 5, "max": 9},
                    "moderate": {"min": 10, "max": 14},
                    "moderate_severe": {"min": 15, "max": 19},
                    "severe": {"min": 20, "max": 27}
                },
                "GAD7": {
                    "normal": {"min": 0, "max": 4},
                    "mild": {"min": 5, "max": 9},
                    "moderate": {"min": 10, "max": 14},
                    "severe": {"min": 15, "max": 21}
                }
            }
        }

    def generate_candidates(self, profile: Dict[str, float]) -> List[Dict]:
        candidates = []
        for rule in self.rules.get("rules", []):
            conditions = rule.get("conditions", {})
            all_conditions_met = True
            for key, threshold in conditions.items():
                if key.endswith("_min"):
                    base_key = key.replace("_min", "")
                    if profile.get(base_key, 0) < threshold:
                        all_conditions_met = False
                        break
                elif key.endswith("_max"):
                    base_key = key.replace("_max", "")
                    if profile.get(base_key, 0) > threshold:
                        all_conditions_met = False
                        break
                else:
                    if profile.get(key, 0) < threshold:
                        all_conditions_met = False
                        break
            if all_conditions_met:
                evidence = rule.get("evidence_template", "")
                for k, v in profile.items():
                    evidence = evidence.replace("{" + k + "}", str(v))
                candidates.append({
                    "diagnosis": rule.get("diagnosis"),
                    "label": rule.get("label", rule.get("diagnosis")),
                    "confidence": rule.get("base_confidence", 0.5),
                    "evidence": evidence,
                    "conditions": conditions
                })
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates

    def eliminative(self, candidates: List[Dict], profile: Dict[str, float]) -> List[Dict]:
        results = []
        for candidate in candidates:
            diagnosis = candidate["diagnosis"]
            exclusion_rule = self.exclusion_rules.get(diagnosis, {})
            exclude_if = exclusion_rule.get("exclude_if", {})

            should_exclude = False
            exclude_reason = ""
            for key, condition in exclude_if.items():
                if isinstance(condition, str):
                    if condition.startswith("<= "):
                        threshold = float(condition.replace("<= ", ""))
                        if profile.get(key, 0) <= threshold:
                            should_exclude = True
                            exclude_reason = f"{key}{condition}（当前值{profile.get(key, 0)}）"
                            break
                    elif condition.startswith(">= "):
                        threshold = float(condition.replace(">= ", ""))
                        if profile.get(key, 0) >= threshold:
                            should_exclude = True
                            exclude_reason = f"{key}{condition}（当前值{profile.get(key, 0)}）"
                            break
                    elif condition.startswith("> "):
                        threshold = float(condition.replace("> ", ""))
                        if profile.get(key, 0) > threshold:
                            should_exclude = True
                            exclude_reason = f"{key}{condition}（当前值{profile.get(key, 0)}）"
                            break
                    elif condition.startswith("< "):
                        threshold = float(condition.replace("< ", ""))
                        if profile.get(key, 0) < threshold:
                            should_exclude = True
                            exclude_reason = f"{key}{condition}（当前值{profile.get(key, 0)}）"
                            break
                    elif condition.startswith("== "):
                        threshold = float(condition.replace("== ", ""))
                        if profile.get(key, 0) == threshold:
                            should_exclude = True
                            exclude_reason = f"{key}{condition}（当前值{profile.get(key, 0)}）"
                            break

            result = candidate.copy()
            result["status"] = "excluded" if should_exclude else "supported"
            result["exclude_reason"] = exclude_reason
            results.append(result)
        return results

    def verifier(self, results: List[Dict]) -> Optional[Dict]:
        valid = [item for item in results if item["status"] == "supported"]
        if not valid:
            return None
        best = max(valid, key=lambda x: x["confidence"])
        return {
            "diagnosis": best["diagnosis"],
            "label": best.get("label", best["diagnosis"]),
            "confidence": best["confidence"],
            "evidence": best["evidence"],
            "status": "confirmed"
        }

    def get_severity(self, profile: Dict[str, float]) -> Dict[str, str]:
        severity_result = {}
        for scale, thresholds in self.severity_thresholds.items():
            score = profile.get(scale, 0)
            found = False
            for level, range_info in thresholds.items():
                min_val = range_info.get("min", 0)
                max_val = range_info.get("max", float('inf'))
                if min_val <= score <= max_val:
                    severity_result[scale] = level
                    found = True
                    break
            if not found:
                severity_result[scale] = "unknown"
        return severity_result

    def get_severity_level(self, severity: Dict[str, str]) -> int:
        if any(v == "severe" for v in severity.values()):
            return 3
        if any(v in ("moderate", "moderate_severe") for v in severity.values()):
            return 2
        return 1

    def _criteria(self, disorder: str, profile: Dict[str, float]):
        met = []
        detail = []
        for code, name, cond, basis in DSM5_CRITERIA_RULES.get(disorder, []):
            ok = bool(cond(profile))
            if ok:
                met.append(code)
            detail.append({"criterion": f"{code} {name}", "met": ok, "evidence": basis})
        return met, detail

    def run(self, profile: Dict[str, float], safety_context: Optional[Dict] = None) -> Dict[str, Any]:
        candidates = self.generate_candidates(profile)
        checked = self.eliminative(candidates, profile)
        severity = self.get_severity(profile)
        severity_level = self.get_severity_level(severity)

        supported = [c for c in checked if c["status"] == "supported"]
        excluded = [c for c in checked if c["status"] == "excluded"]
        supported.sort(key=lambda x: x["confidence"], reverse=True)

        primary = None
        secondary = []
        if supported:
            best = supported[0]
            scale = DISORDER_SEVERITY_SCALE.get(best["diagnosis"])
            primary = {
                "disorder": best["diagnosis"],
                "confidence": best["confidence"],
                "severity": severity.get(scale, "unknown"),
                "evidence": best["evidence"],
            }
            met, detail = self._criteria(best["diagnosis"], profile)
            primary["dsm5_criteria_met"] = met
            primary["dsm5_criteria_detail"] = detail
            for other in supported[1:]:
                secondary.append({
                    "disorder": other["diagnosis"],
                    "confidence": other["confidence"],
                    "evidence": other["evidence"],
                })

        ruled_out = []
        for e in excluded:
            ruled_out.append({
                "disorder": e["diagnosis"],
                "reason": e.get("exclude_reason") or "不满足诊断标准",
                "evidence_ref": "DSM-5/ICD-11 排除标准",
            })

        verifier_score = primary["confidence"] if primary else 0.0
        if verifier_score >= 0.8:
            verifier_routing = "pass"
        elif verifier_score >= 0.5:
            verifier_routing = "conditional"
        else:
            verifier_routing = "reject"

        evidence_refs = []
        if primary:
            evidence_refs.append({
                "type": "DSM-5",
                "disorder": primary["disorder"],
                "criteria_met": f"{','.join(primary['dsm5_criteria_met'])} ({len(primary['dsm5_criteria_met'])}条满足)",
                "source_doc": "DSM-5_TR_zh_v1.0",
            })
            scale = DISORDER_SEVERITY_SCALE.get(primary["disorder"])
            if scale and scale in profile:
                evidence_refs.append({
                    "type": "scale",
                    "disorder": primary["disorder"],
                    "source": scale,
                    "threshold": f"score={profile[scale]}, severity={primary['severity']}",
                })
        for s in secondary:
            evidence_refs.append({
                "type": "scale",
                "disorder": s["disorder"],
                "source": DISORDER_SEVERITY_SCALE.get(s["disorder"], ""),
                "threshold": f"confidence={s['confidence']}",
            })
        for r in ruled_out:
            evidence_refs.append({
                "type": "exclusion",
                "disorder": r["disorder"],
                "reason": r["reason"],
                "source": "DSM-5/ICD-11 排除标准",
            })

        flags = list(safety_context.get("flags", [])) if safety_context else []
        if any(s["disorder"] == "Insomnia" for s in secondary):
            flags.append("insomnia_comorbid")
        if primary and primary["disorder"] == "Bipolar":
            flags.append("medical_referral")

        summary_parts = []
        if primary:
            summary_parts.append(
                f"初步印象为{primary['disorder']}（置信度{primary['confidence']}，"
                f"Verifier={verifier_score}，{verifier_routing}）"
            )
        if secondary:
            summary_parts.append("共病：" + "、".join(s["disorder"] for s in secondary))
        if ruled_out:
            summary_parts.append("排除：" + "、".join(r["disorder"] for r in ruled_out))

        return {
            "diagnosis": {
                "primary": primary,
                "secondary": secondary,
                "ruled_out": ruled_out,
            },
            "verifier_score": verifier_score,
            "verifier_routing": verifier_routing,
            "severity": severity,
            "severity_level": severity_level,
            "candidates": candidates,
            "checked": checked,
            "evidence_refs": evidence_refs,
            "confidence": verifier_score,
            "flags": flags,
            "summary": "；".join(summary_parts) or "未获得可确认的诊断结论，需补充信息。",
        }
