from __future__ import annotations

from typing import Any

from layers.layer3.diagnosis.schemas import (
    CandidateGeneration,
    DiagnosticCandidate,
    EliminativeAssessment,
    RewardModelAssessment,
    SUPPORTED_DISORDERS,
)


def _known_ref(profile: dict[str, Any]) -> str:
    items = profile.get("PHQ9_items") or {}
    if "item2" in items:
        return f"PHQ9_item2={items['item2']}"
    for key in ("PHQ9", "GAD7", "PCL5", "Y_BOCS", "ISI", "MDQ"):
        if key in profile:
            return f"{key}={profile[key]}"
    nodes = profile.get("evidence_nodes") or []
    if nodes:
        return str(nodes[0]["evidence_id"])
    raise ValueError("受控L3模型需要至少一个可引用的输入证据")


class ControlledCLINICRReasoner:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.evaluate_calls: list[str] = []

    def generate_candidates(self, profile: dict[str, Any]) -> CandidateGeneration:
        self.generate_calls.append(profile)
        ref = _known_ref(profile)
        return CandidateGeneration.model_validate({
            "considered_disorders": list(SUPPORTED_DISORDERS),
            "candidates": [
                {
                    "disorder": "D013",
                    "label": "重性抑郁障碍",
                    "symptom_match": {"matched": 5, "total": 9},
                    "core_evidence_summary": "低落、兴趣减退及相关量表证据",
                    "evidence_refs": [ref],
                    "confidence": 0.75,
                },
                {
                    "disorder": "D011",
                    "label": "广泛性焦虑障碍",
                    "symptom_match": {"matched": 2, "total": 6},
                    "core_evidence_summary": "存在需要继续核对的焦虑线索",
                    "evidence_refs": [ref],
                    "confidence": 0.45,
                },
                {
                    "disorder": "D020",
                    "label": "适应障碍伴抑郁情绪",
                    "symptom_match": {"matched": 2, "total": 5},
                    "core_evidence_summary": "存在应激相关线索",
                    "evidence_refs": [ref],
                    "confidence": 0.35,
                },
                {
                    "disorder": "D009",
                    "label": "双相Ⅱ型（抑郁相）",
                    "symptom_match": {"matched": 1, "total": 7},
                    "core_evidence_summary": "作为重要排除诊断保留",
                    "evidence_refs": [ref],
                    "confidence": 0.15,
                },
            ],
        })

    def evaluate_candidate(
        self,
        profile: dict[str, Any],
        candidate: DiagnosticCandidate,
    ) -> EliminativeAssessment:
        self.evaluate_calls.append(candidate.disorder)
        ref = _known_ref(profile)
        if candidate.disorder == "D013":
            items = profile.get("PHQ9_items") or {}
            required = {"item1", "item2", "item3", "item4", "item6"}
            if not required <= set(items):
                return EliminativeAssessment.model_validate({
                    "disorder": "D013",
                    "criteria_detail": [{
                        "code": "A1",
                        "name": "心境抑郁",
                        "status": "pending",
                        "rationale": "缺少单题和患者叙述证据",
                        "evidence_refs": [],
                    }],
                    "conclusion": "watch",
                    "conclusion_reason": "证据不足，暂列待观察",
                    "missing_dimensions": [{
                        "field": "phq9_items.item2",
                        "purpose": "核对心境抑郁条目",
                        "suggested_question": "最近两周情绪低落出现得有多频繁？",
                    }],
                })

            mapping = {
                "A1": ("心境抑郁", "item2"),
                "A2": ("兴趣减退", "item1"),
                "A3": ("食欲/体重变化", "item5"),
                "A4": ("睡眠障碍", "item3"),
                "A5": ("精神运动性改变", "item8"),
                "A6": ("疲劳/精力减退", "item4"),
                "A7": ("无价值感/过度自责", "item6"),
                "A8": ("注意力减退", "item7"),
                "A9": ("自杀意念", "item9"),
            }
            criteria = []
            for code, (name, item_key) in mapping.items():
                score = items.get(item_key, 0)
                threshold = 1 if item_key == "item9" else 2
                criteria.append({
                    "code": code,
                    "name": name,
                    "status": "met" if score >= threshold else "not_met",
                    "rationale": "依据PHQ-9单题受控判定",
                    "evidence_refs": [f"PHQ9_{item_key}={score}"],
                })
            return EliminativeAssessment.model_validate({
                "disorder": "D013",
                "criteria_detail": criteria,
                "conclusion": "meets_criteria",
                "conclusion_reason": "满足5/9条且包含A1和A2",
                "missing_dimensions": [],
            })

        if candidate.disorder in {"D011", "D020"}:
            return EliminativeAssessment.model_validate({
                "disorder": candidate.disorder,
                "criteria_detail": [{
                    "code": "C1",
                    "name": "核心条件",
                    "status": "pending",
                    "rationale": "当前画像不足以确认核心条件",
                    "evidence_refs": [],
                }],
                "conclusion": "watch",
                "conclusion_reason": "部分线索存在但不足以形成诊断",
                "missing_dimensions": [{
                    "field": f"{candidate.disorder}.core_criterion",
                    "purpose": "确认核心诊断条件",
                    "suggested_question": "这些表现通常在什么情境下发生并持续多久？",
                }],
            })

        return EliminativeAssessment.model_validate({
            "disorder": candidate.disorder,
            "criteria_detail": [{
                "code": "CORE",
                "name": "核心诊断条件",
                "status": "not_met",
                "rationale": "输入证据明确不支持核心条件",
                "evidence_refs": [ref],
            }],
            "conclusion": "ruled_out",
            "conclusion_reason": "不满足核心诊断标准",
            "missing_dimensions": [],
        })


class ControlledRewardModelVerifier:
    def __init__(self, score: float = 0.87) -> None:
        self.score = score
        self.calls: list[dict[str, Any]] = []

    def score_reasoning(
        self,
        profile: dict[str, Any],
        candidate_without_confidence: dict[str, Any],
        assessment: EliminativeAssessment,
        differential_context: list[dict[str, Any]],
    ) -> RewardModelAssessment:
        self.calls.append({
            "candidate": candidate_without_confidence,
            "assessment": assessment.model_dump(mode="json"),
            "differential_context": differential_context,
        })
        ref = _known_ref(profile)
        needs_more = self.score < 0.8
        return RewardModelAssessment.model_validate({
            "score": self.score,
            "dimensions": [
                {
                    "dimension": dimension,
                    "score": self.score,
                    "reason": "受控终验维度评分",
                    "evidence_refs": [ref] if dimension == "evidence_sufficiency" else [],
                }
                for dimension in (
                    "evidence_sufficiency",
                    "logic_coherence",
                    "differential_coverage",
                    "no_reasoning_leaps",
                )
            ],
            "fail_reasons": (
                ["推理质量低于最低门槛"] if self.score < 0.5 else []
            ),
            "missing_dimensions": ([{
                "field": "diagnostic_context.additional_detail",
                "purpose": "补充终验所需的诊断信息",
                "suggested_question": "这些症状持续多久，并造成了哪些具体影响？",
            }] if needs_more else []),
        })


def controlled_clinicr(score: float = 0.87):
    from layers.layer3.diagnosis.clinicr import CLINICR
    from layers.layer3.diagnosis.verifier import IndependentVerifier

    reasoner = ControlledCLINICRReasoner()
    verifier_model = ControlledRewardModelVerifier(score=score)
    engine = CLINICR(
        reasoner=reasoner,
        verifier=IndependentVerifier(model=verifier_model),
    )
    return engine, reasoner, verifier_model
