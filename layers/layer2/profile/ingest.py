from __future__ import annotations

from typing import Any

from layers.layer2.evidence import EvidenceRegistry
from layers.layer2.schemas import (
    CorrectionCandidate,
    L2SessionState,
    ProfileFact,
    SemanticAnalysis,
    SemanticCandidate,
    SymptomEvidence,
)


from shared.disorder_catalog import SYMPTOMS

ALLOWED_SYMPTOMS = {
    "depressed_mood",
    "anhedonia",
    "appetite_or_weight_change",
    "insomnia",
    "fatigue",
    "psychomotor_change",
    "worthlessness",
    "concentration_difficulty",
    "anxiety",
    "trauma_related",
    "obsession_compulsion",
    "elevated_mood",
    "passive_si",
}
ALLOWED_SYMPTOMS.update(SYMPTOMS)

ALLOWED_STRUCTURED_FIELDS = {
    "symptom_duration_days",
    "functional_impairment",
    "mania_or_hypomania_history",
    "substance_or_medical_cause",
    "differential_diagnoses_assessed",
    "evidence_conflicts",
    "diagnostic_details",
    "relapse_count",
    "rhythm_disturbance",
    "exposure_avoidant",
    "ritual_behaviors",
    "insomnia_chief_complaint",
    "chronic_months",
    "interpersonal_conflicts",
    "interpersonal_impairment",
    "single_focus",
    "psychotic_symptoms",
    "substance_active",
    "severe_dissociation",
    "motivation_importance",
    "motivation_confidence",
    "motivation_readiness",
}


class ProfileIngestor:
    CORRECTABLE_LISTS = {
        "emotions",
        "cognitive_patterns",
        "patient_goals",
        "key_narratives",
        "functional_impairments",
        "focus_topics",
    }

    def apply_corrections(
        self,
        state: L2SessionState,
        *,
        analysis: SemanticAnalysis,
        patient_text: str,
        message_id: str,
    ) -> list[str]:
        evidence_ids: list[str] = []
        for correction in analysis.corrections:
            if correction.evidence_quote not in patient_text:
                continue
            parent_ids = self._remove_target(state, correction)
            if parent_ids is None:
                continue
            node = EvidenceRegistry.add_patient_evidence(
                state,
                message_id=message_id,
                quote=correction.evidence_quote,
                target_path=f"corrections.{correction.target_path}",
                parent_ids=parent_ids,
            )
            evidence_ids.append(node.evidence_id)
        return evidence_ids

    def ingest(
        self,
        state: L2SessionState,
        *,
        analysis: SemanticAnalysis,
        patient_text: str,
        message_id: str,
    ) -> list[str]:
        evidence_ids: list[str] = []
        for candidate in analysis.symptoms:
            if candidate.value not in ALLOWED_SYMPTOMS:
                continue
            node = self._evidence(
                state, candidate, patient_text, message_id,
                f"symptoms.{candidate.value}",
            )
            if node is None:
                continue
            evidence_ids.append(node.evidence_id)
            existing = state.symptoms.get(candidate.value)
            if existing:
                if node.evidence_id not in existing.sources:
                    existing.sources.append(node.evidence_id)
                if candidate.evidence_quote not in existing.patient_quotes:
                    existing.patient_quotes.append(candidate.evidence_quote)
                existing.attributes.update(candidate.attributes)
            else:
                state.symptoms[candidate.value] = SymptomEvidence(
                    symptom=candidate.value,
                    sources=[node.evidence_id],
                    patient_quotes=[candidate.evidence_quote],
                    attributes=candidate.attributes,
                )

        collections = (
            ("emotions", analysis.emotions),
            ("cognitive_patterns", analysis.cognitive_patterns),
            ("patient_goals", analysis.patient_goals),
            ("key_narratives", analysis.key_narratives),
            ("functional_impairments", analysis.functional_impairments),
            ("focus_topics", analysis.focus_topics),
            ("change_talk", analysis.change_talk),
            ("sustain_talk", analysis.sustain_talk),
        )
        for target, candidates in collections:
            destination: list[ProfileFact] = getattr(state, target)
            for candidate in candidates:
                existing_index = next(
                    (index for index, item in enumerate(destination)
                     if item.value == candidate.value),
                    None,
                )
                target_index = (
                    existing_index if existing_index is not None else len(destination)
                )
                node = self._evidence(
                    state, candidate, patient_text, message_id,
                    f"{target}.{target_index}",
                )
                if node is None:
                    continue
                evidence_ids.append(node.evidence_id)
                if existing_index is not None:
                    existing = destination[existing_index]
                    if node.evidence_id not in existing.sources:
                        existing.sources.append(node.evidence_id)
                    if candidate.evidence_quote not in existing.patient_quotes:
                        existing.patient_quotes.append(candidate.evidence_quote)
                else:
                    destination.append(ProfileFact(
                        value=candidate.value,
                        sources=[node.evidence_id],
                        patient_quotes=[candidate.evidence_quote],
                    ))

        for candidate in analysis.structured_fields:
            if candidate.field not in ALLOWED_STRUCTURED_FIELDS:
                continue
            if candidate.evidence_quote not in patient_text:
                continue
            if not self._valid_structured_value(candidate.field, candidate.value):
                continue
            node = EvidenceRegistry.add_patient_evidence(
                state,
                message_id=message_id,
                quote=candidate.evidence_quote,
                target_path=f"diagnostic_context.{candidate.field}",
            )
            evidence_ids.append(node.evidence_id)
            self._write_structured(state, candidate.field, candidate.value, node.evidence_id)
        return evidence_ids

    @staticmethod
    def _evidence(
        state: L2SessionState,
        candidate: SemanticCandidate,
        patient_text: str,
        message_id: str,
        target_path: str,
    ):
        if candidate.evidence_quote not in patient_text:
            return None
        return EvidenceRegistry.add_patient_evidence(
            state,
            message_id=message_id,
            quote=candidate.evidence_quote,
            target_path=target_path,
        )

    @staticmethod
    def _valid_structured_value(field: str, value: Any) -> bool:
        if field in {
            "symptom_duration_days",
            "relapse_count",
            "chronic_months",
            "interpersonal_conflicts",
        }:
            return (
                isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value <= 36500
            )
        if field in {
            "functional_impairment",
            "mania_or_hypomania_history",
            "substance_or_medical_cause",
            "rhythm_disturbance",
            "exposure_avoidant",
            "ritual_behaviors",
            "insomnia_chief_complaint",
            "interpersonal_impairment",
            "single_focus",
            "psychotic_symptoms",
            "substance_active",
            "severe_dissociation",
        }:
            return isinstance(value, bool)
        if field == "differential_diagnoses_assessed":
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        if field == "diagnostic_details":
            return isinstance(value, dict) and all(isinstance(key, str) for key in value)
        if field == "evidence_conflicts":
            return isinstance(value, list) and all(isinstance(item, dict) for item in value)
        if field.startswith("motivation_"):
            return (
                isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value <= 10
            )
        return False

    @staticmethod
    def _write_structured(
        state: L2SessionState, field: str, value: Any, evidence_id: str
    ) -> None:
        if field.startswith("motivation_"):
            name = field.removeprefix("motivation_")
            setattr(state.motivation, name, value)
            if evidence_id not in state.motivation.sources:
                state.motivation.sources.append(evidence_id)
            return
        if field == "diagnostic_details":
            state.diagnostic_context.diagnostic_details.update(value)
        else:
            setattr(state.diagnostic_context, field, value)
        state.diagnostic_context.sources.setdefault(field, []).append(evidence_id)

    def _remove_target(
        self, state: L2SessionState, correction: CorrectionCandidate
    ) -> list[str] | None:
        target = correction.target_path
        if target.startswith("symptoms."):
            symptom = target.split(".", 1)[1]
            if symptom not in ALLOWED_SYMPTOMS:
                return None
            existing = state.symptoms.pop(symptom, None)
            return list(existing.sources) if existing else []
        if target in self.CORRECTABLE_LISTS:
            existing = list(getattr(state, target))
            setattr(state, target, [])
            return [source for fact in existing for source in fact.sources]
        if target.startswith("motivation."):
            field = target.split(".", 1)[1]
            if field not in {"importance", "confidence", "readiness"}:
                return None
            parent_ids = list(state.motivation.sources)
            setattr(state.motivation, field, None)
            return parent_ids
        if target.startswith("diagnostic_context."):
            field = target.split(".", 1)[1]
            if field not in ALLOWED_STRUCTURED_FIELDS or field.startswith("motivation_"):
                return None
            existing_sources = state.diagnostic_context.sources.pop(field, [])
            setattr(state.diagnostic_context, field, None)
            return existing_sources
        if target.startswith("scale_results."):
            scale = target.split(".", 1)[1]
            existing = state.scale_results.get(scale)
            if existing is None:
                return []
            parent_ids = list(existing.evidence_ids)
            try:
                state.scale_results[scale] = type(existing)(scale_name=scale)
            except ValueError:
                return None
            if state.pending_scale_name == scale:
                state.pending_scale_name = None
            return parent_ids
        return None
