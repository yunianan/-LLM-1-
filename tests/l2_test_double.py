from __future__ import annotations

import re
from typing import Any

from layers.layer2.schemas import (
    CorrectionCandidate,
    SemanticAnalysis,
    SemanticCandidate,
    StageDecision,
    StructuredFieldCandidate,
)


class ControlledL2Model:
    """自动化测试专用受控模型；生产代码不会默认使用它。"""

    def __init__(self) -> None:
        self.analyze_calls = 0
        self.generate_calls = 0

    def analyze(self, patient_text: str, context: dict[str, Any]) -> SemanticAnalysis:
        self.analyze_calls += 1
        expected = context.get("expected_input")
        quote = patient_text.strip()
        structured: list[StructuredFieldCandidate] = []

        if any(term in quote for term in ("暂停一下", "先暂停", "暂时停止")):
            return SemanticAnalysis(input_type="pause")

        if expected == "scale_consent":
            if any(term in quote for term in ("同意", "愿意", "可以")):
                value = True
            elif any(term in quote for term in ("拒绝", "不愿意", "不同意")):
                value = False
            else:
                value = "unknown"
            structured.append(StructuredFieldCandidate(
                field="scale_consent", value=value, evidence_quote=quote
            ))
            return SemanticAnalysis(input_type="scale_consent", structured_fields=structured)

        if expected == "scale_answer" or context.get("pending_scale_item"):
            return SemanticAnalysis(input_type="scale_answer")

        if expected == "profile_confirmation":
            decision = None
            if any(term in quote for term in ("准确", "确认", "继续", "没问题")):
                decision = "continue"
            elif any(term in quote for term in ("修改", "不准确", "有误")):
                decision = "revise"
            elif any(term in quote for term in ("暂停", "不继续")):
                decision = "pause"
            if decision:
                structured.append(StructuredFieldCandidate(
                    field="confirmation_decision",
                    value=decision,
                    evidence_quote=quote,
                ))
            return SemanticAnalysis(
                input_type="profile_confirmation", structured_fields=structured
            )

        symptoms = []
        symptom_terms = {
            "depressed_mood": ("低落", "沮丧"),
            "anhedonia": ("没兴趣", "没有兴趣", "提不起兴趣"),
            "insomnia": ("睡不好", "失眠", "睡不着"),
            "fatigue": ("没有精力", "没力气", "疲倦"),
            "worthlessness": ("让人失望", "没用", "自责"),
            "concentration_difficulty": ("注意力", "无法集中"),
            "anxiety": ("焦虑", "担心", "紧张"),
            "trauma_related": ("噩梦", "闪回", "惊吓"),
            "obsession_compulsion": ("反复想", "反复做", "仪式"),
            "elevated_mood": ("异常兴奋", "不睡不累", "说话很快"),
        }
        for name, terms in symptom_terms.items():
            if any(term in quote for term in terms):
                symptoms.append(SemanticCandidate(value=name, evidence_quote=quote))

        focus = []
        if expected == "focus_selection" or any(
            term in quote for term in ("想先聊", "先从", "最想先")
        ):
            focus.append(SemanticCandidate(value=quote, evidence_quote=quote))

        goals = []
        if expected == "patient_goals" or any(term in quote for term in ("目标", "希望")):
            goals.append(SemanticCandidate(value=quote, evidence_quote=quote))
        corrections = []
        if "目标不是" in quote and "而是" in quote:
            corrections.append(CorrectionCandidate(
                target_path="patient_goals",
                action="replace",
                evidence_quote=quote,
            ))

        if expected == "motivation":
            scores = [int(value) for value in re.findall(r"([0-9]|10)\s*分", quote)]
            names = ["importance", "confidence", "readiness"]
            for name, value in zip(names, scores):
                structured.append(StructuredFieldCandidate(
                    field=f"motivation_{name}", value=value, evidence_quote=quote
                ))

        duration_days = None
        day_match = re.search(r"(\d+)\s*天", quote)
        week_match = re.search(r"(\d+)\s*[个]?周", quote)
        month_match = re.search(r"(\d+)\s*[个]?月", quote)
        if day_match:
            duration_days = int(day_match.group(1))
        elif week_match:
            duration_days = int(week_match.group(1)) * 7
        elif month_match:
            duration_days = int(month_match.group(1)) * 30
        elif "三周" in quote:
            duration_days = 21
        elif "两周" in quote:
            duration_days = 14
        elif "两个月" in quote:
            duration_days = 60
        if duration_days is not None:
            structured.append(StructuredFieldCandidate(
                field="symptom_duration_days",
                value=duration_days,
                evidence_quote=quote,
            ))

        if expected == "functional_impairment" or any(
            term in quote for term in ("影响了工作", "影响学习", "影响生活")
        ):
            structured.append(StructuredFieldCandidate(
                field="functional_impairment",
                value=not any(term in quote for term in ("没有影响", "尚未影响")),
                evidence_quote=quote,
            ))
        if expected == "mania_or_hypomania_history" or "少睡却不困" in quote:
            structured.append(StructuredFieldCandidate(
                field="mania_or_hypomania_history",
                value=not any(term in quote for term in ("从未", "没有")),
                evidence_quote=quote,
            ))
        if expected == "substance_or_medical_cause" or any(
            term in quote for term in ("药物", "物质", "身体疾病")
        ):
            structured.append(StructuredFieldCandidate(
                field="substance_or_medical_cause",
                value=not any(term in quote for term in ("并非", "不是", "没有")),
                evidence_quote=quote,
            ))
        if expected == "psychotic_symptoms" or any(
            term in quote for term in ("别人听不到", "被监视", "分辨现实")
        ):
            structured.append(StructuredFieldCandidate(
                field="psychotic_symptoms",
                value=not any(term in quote for term in ("没有", "从未")),
                evidence_quote=quote,
            ))
        if expected == "differential_diagnoses_assessed" or "双相障碍" in quote:
            structured.append(StructuredFieldCandidate(
                field="differential_diagnoses_assessed",
                value=["Bipolar", "Adjustment"],
                evidence_quote=quote,
            ))
        if expected == "evidence_conflicts" or "量表结果" in quote:
            conflicts = [] if any(term in quote for term in ("一致", "没有矛盾")) else [
                {"description": quote, "resolved": False}
            ]
            structured.append(StructuredFieldCandidate(
                field="evidence_conflicts", value=conflicts, evidence_quote=quote
            ))

        scales = []
        if any(item.value in {"depressed_mood", "anhedonia"} for item in symptoms):
            scales.append("PHQ-9")
        if any(item.value == "anxiety" for item in symptoms):
            scales.append("GAD-7")
        if any(item.value == "trauma_related" for item in symptoms):
            scales.append("PCL-5")
        if any(item.value == "obsession_compulsion" for item in symptoms):
            scales.append("Y-BOCS")
        if any(item.value == "insomnia" for item in symptoms) and not scales:
            scales.append("ISI")
        if any(item.value == "elevated_mood" for item in symptoms):
            scales.append("MDQ")

        input_type = "free_narrative"
        if focus:
            input_type = "focus_selection"
        elif goals:
            input_type = "goal_statement"
        elif expected == "motivation":
            input_type = "motivation_rating"

        return SemanticAnalysis(
            input_type=input_type,
            symptoms=symptoms,
            emotions=(
                [SemanticCandidate(value="低落", evidence_quote=quote)]
                if any(item.value == "depressed_mood" for item in symptoms) else []
            ),
            key_narratives=(
                [SemanticCandidate(value=quote, evidence_quote=quote)] if symptoms else []
            ),
            focus_topics=focus,
            patient_goals=goals,
            possible_scale_domains=scales,
            structured_fields=structured,
            corrections=corrections,
        )

    def generate(self, decision: StageDecision, context: dict[str, Any]) -> str:
        self.generate_calls += 1
        if decision.required_question:
            return "谢谢你说明刚才的情况。" + decision.required_question
        if "是否愿意" in decision.response_goal:
            return "这与你刚才提到的困扰有关。我想用一个简短量表更系统地了解情况，你愿意吗？"
        if "准确、修改或暂停" in decision.response_goal:
            return "我已经把你刚才提供的信息整理好了。请确认信息准确并继续，或告诉我需要修改，也可以选择暂停。"
        return "我听见了你刚才说的这些。" + decision.response_goal + "？"
