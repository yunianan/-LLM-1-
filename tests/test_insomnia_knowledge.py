import json
from pathlib import Path

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.llm import client
from shared.disorder_catalog import SUPPLEMENT_ROOT, candidate_knowledge
from tests.test_diagnostic_catalog import NodeReasoner
from tests.test_layer3_clinicr import _profile
from tests.tree_test_support import node_payload


def _resolve(ref):
    filename, pointer = ref.split("#", 1)
    path = Path(__file__).resolve().parents[1] / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    for part in pointer.lstrip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def test_insomnia_conditions_preserve_frequency_duration_and_exclusions():
    knowledge = candidate_knowledge("Insomnia")
    required = knowledge["criteria"]["required_criteria"]
    assert required["insomnia_symptoms"]["min_count"] == 1
    assert set(required["insomnia_symptoms"]["symptom_pool"]) == {"S085", "S086", "S087"}
    assert _resolve(knowledge["criterion_source_refs"]["D"]) == "3 months"
    assert "3 nights per week" in _resolve(knowledge["criterion_source_refs"]["C"])
    assert "OR" in _resolve(knowledge["criterion_source_refs"]["B"])
    assert "sufficient opportunity" in _resolve(knowledge["criterion_source_refs"]["E"])
    assert "sleep-wake disorder" in _resolve(knowledge["criterion_source_refs"]["F"])
    assert "Substance or medication" in _resolve(knowledge["criterion_source_refs"]["G"])
    assert "do not sufficiently explain" in _resolve(knowledge["criterion_source_refs"]["H"])
    for ref in [knowledge["source_ref"], *knowledge["criterion_source_refs"].values()]:
        assert _resolve(ref)
        assert "paper" not in ref and "arxiv" not in ref
    assert required["functional_impairment_required"] is True
    assert "impairment alone is not mandatory" in knowledge["events"]["functional_impairment_required"]["description"]


def test_supplement_uses_mental_kg_file_shapes_and_existing_differential_ids():
    knowledge = candidate_knowledge("Insomnia")
    assert json.loads((SUPPLEMENT_ROOT / "disorder.json").read_text())["Insomnia"] == {"name": "Insomnia Disorder"}
    for item in knowledge["differential"]["Insomnia"]:
        assert item["differential_diagnosis"] in {"D011", "D013"}
        assert set(item["rules"]) == {"Insomnia", item["differential_diagnosis"]}
        assert "alone does not exclude" in item["rules"]["Insomnia"][0]


def test_generation_and_elimination_receive_supplement_not_placeholder(monkeypatch):
    reasoner = NodeReasoner("Insomnia")
    candidate = reasoner.generate_candidates(_profile()).candidates[0]
    prompts = []
    def fake_ask(prompt, **kwargs):
        prompts.append(prompt)
        if "只执行 Step 1" in prompt:
            return json.dumps(node_payload(prompt, ["Insomnia"], candidate.evidence_refs[0]))
        return reasoner.evaluate_candidate(_profile(), candidate).model_dump_json()
    monkeypatch.setattr(client, "ask_llm_stream", fake_ask)
    model = client.APICLINICRReasoner(model="offline-test")
    profile = _profile() | {client.ALLOWED_EVIDENCE_REFS_KEY: sorted(CLINICR._known_evidence_refs(_profile()))}
    model.generate_candidates(profile)
    model.evaluate_candidate(profile, candidate)
    for prompt in [p for p in prompts if "3 nights per week" in p]:
        assert "3 nights per week" in prompt
        assert "3 months" in prompt
        assert "S085" in prompt
        assert "knowledge/dsm5_supplement/EN/diagnostic_criteria.json#/Insomnia" in prompt
    assert any("3 nights per week" in prompt for prompt in prompts[:-1])
    assert "原图谱作者提供或审核" in prompts[-1]


def test_insomnia_report_criterion_source_points_to_specific_json_item():
    profile = _profile()
    reasoner = NodeReasoner("Insomnia")
    candidate = reasoner.generate_candidates(profile).candidates[0]
    assessment = reasoner.evaluate_candidate(profile, candidate)
    assessment.criteria_detail[0].code = "C"
    result = CLINICR._result(
        candidates=[candidate], assessments=[assessment], primary=None,
        provisional_primary=None, secondary=[], under_observation=[], ruled_out=[],
        severity={}, severity_level=1,
        verifier={"score": 0, "decision": "rejected", "routing": "return_to_l2",
                  "checks": [], "fail_reasons": [], "missing_dimensions": []},
        safety_context=None,
    )
    refs = [item for item in result["evidence_refs"] if item["type"] == "DSM-5-criterion"]
    assert len(refs) == 1
    assert refs[0]["source_doc"].endswith("/additional_requirements/2")
    assert "3 nights per week" in _resolve(refs[0]["source_doc"])
