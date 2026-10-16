"""从正式病例派生依据索引；编号复用原有 fact_id/symptom_id。"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from patient_agent.schemas import PatientScenario

PROMPT_VERSION = "patient-evidence-v1"


class FactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_id: str
    scenario_id: str
    scenario_version: str
    value: str
    status: Literal["present", "denied", "non_factual"]
    source_values: dict[str, object]


class AnswerSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: Literal["factual", "non_factual"]
    fact_ids: list[str]


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[AnswerSegment] = Field(min_length=1)


def evidence_bundle(scenario: PatientScenario, fact_id: str) -> list[FactEvidence]:
    """数组位置只作定位；已有稳定业务编号才作为事实 ID。"""
    entries: list[FactEvidence] = []
    def add(key, value, sources, status="present"):
        entries.append(FactEvidence(fact_id=key, value=value, source_values=sources,
            status=status, scenario_id=scenario.scenario_id, scenario_version=scenario.version))

    if not fact_id:
        replies = [(v, {f"disclosure.non_fact_replies[{i}]": v}) for i, v in enumerate(scenario.disclosure.non_fact_replies)]
        replies += [(v, {f"disclosure.refusal_utterances[{i}]": v}) for i, v in enumerate(scenario.disclosure.refusal_utterances)]
        replies += [("你觉得呢？我现在也说不太清楚。", {"policy.neutral.askback": "你觉得呢？我现在也说不太清楚。"}),
                    ("嗯，我在听。", {"policy.neutral.filler": "嗯，我在听。"})]
        for i, text in enumerate(scenario.disclosure.refusal_utterances):
            for j, plan in enumerate(scenario.clinical.scale_plans):
                if plan.comment:
                    replies.append((text + f"（{plan.comment}）", {f"disclosure.refusal_utterances[{i}]": text,
                        f"clinical.scale_plans[{j}].comment": plan.comment}))
        for i, (value, sources) in enumerate(replies):
            add(f"neutral:{i}", value, sources, "non_factual")
        return entries
    if fact_id == "opening":
        templates = scenario.disclosure.opening_templates
        for i, value in enumerate(templates):
            add(f"opening:{i}", value, {f"disclosure.opening_templates[{i}]": value})
        for i, value in enumerate(templates):
            add(f"opening:{len(templates)+i}", value + scenario.clinical.chief_complaint,
                {f"disclosure.opening_templates[{i}]": value, "clinical.chief_complaint": scenario.clinical.chief_complaint})
        return entries
    if fact_id.startswith("trajectory:"):
        suffix = fact_id.split(":", 1)[1]
        if suffix.isdigit() and int(suffix) < len(scenario.trajectory.steps):
            value = scenario.trajectory.steps[int(suffix)].utterance
            add(fact_id, value, {f"trajectory.steps[{suffix}].utterance": value})
        return entries

    facts = scenario.revealable_facts()
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("病例存在重复事实编号")
    fact = next((f for f in facts if f.fact_id == fact_id), None)
    if fact is None:
        return []
    sources = {}
    if fact_id.startswith("risk:"):
        field = "denial_utterance" if fact.denial_expected else "utterance"
        sources = {f"risk.{field}": getattr(scenario.risk, field), "risk.level": scenario.risk.level}
    elif fact_id.startswith(("symptom:", "symptom_absent:")):
        sid = fact_id.split(":", 1)[1]
        i, symptom = next((i, s) for i, s in enumerate(scenario.clinical.symptoms) if s.symptom_id == sid)
        field = "denial_utterance" if fact.denial_expected else "utterance"
        sources = {f"clinical.symptoms[{i}].{field}": getattr(symptom, field), f"clinical.symptoms[{i}].present": symptom.present}
    elif fact_id.startswith(("exclusion:", "exclusion_absent:")):
        cid = fact_id.split(":", 1)[1]
        i, item = next((i, s) for i, s in enumerate(scenario.clinical.exclusions) if s.condition_id == cid)
        field = "denial_utterance" if fact.denial_expected else "utterance"
        sources = {f"clinical.exclusions[{i}].{field}": getattr(item, field), f"clinical.exclusions[{i}].present": item.present}
    elif fact_id.startswith("background:"):
        bid = fact_id.split(":", 1)[1]
        i, item = next((i, s) for i, s in enumerate(scenario.background.facts) if s.fact_id == bid)
        sources = {f"background.facts[{i}].content": item.content}
    elif fact_id == "psychology:motivation":
        sources = {"psychology.ambivalence": scenario.psychology.ambivalence, "psychology.goal": scenario.psychology.goal}
    elif fact_id.startswith("psychology:belief:"):
        i = int(fact_id.rsplit(":", 1)[1])
        sources = {f"psychology.core_beliefs[{i}]": scenario.psychology.core_beliefs[i]}
    add(fact_id, fact.denial_utterance if fact.denial_expected else fact.utterance,
        sources, "denied" if fact.denial_expected else "present")
    if fact_id.startswith("symptom:") and symptom.duration != "unknown":
        add(fact_id + ":duration", symptom.duration, {f"clinical.symptoms[{i}].duration": symptom.duration})
    return entries
