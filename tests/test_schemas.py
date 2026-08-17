import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from l4.schemas.review import ReviewResult
from l4.schemas.summary import EvidenceRef, StructuredSummary

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_role_instruction():
    r = RoleInstruction(identity="CBT治疗师", style="温暖")
    assert r.identity == "CBT治疗师"
    with pytest.raises(ValidationError):
        RoleInstruction()  # 缺少必填字段


def test_turn_instruction_force_substage_enum():
    t = TurnInstruction(goal="g", technique="t", force_substage="L4-L2")
    assert t.force_substage == "L4-L2"
    with pytest.raises(ValidationError):
        TurnInstruction(goal="g", technique="t", force_substage="L4-L5")


def test_anchor_therapy_type():
    a = DiagnosticAnchor(primary="MDD", therapy_type="CBT")
    assert a.therapy_type == "CBT"


def test_safety_passport_defaults():
    s = SafetyPassport()
    assert s.risk_level == 1
    assert s.sensitive_topics == []


def test_review_result_passed():
    ok = ReviewResult(harm="pass", boundary="pass", quality="pass")
    assert ok.passed is True
    bad = ReviewResult(harm="fail", boundary="pass", quality="pass",
                       fail_reason="x", fix_instruction="y")
    assert bad.passed is False


def test_l5_context_substage_enum():
    ctx = L5Context(l4_raw_output="你好", l4_substage="L4-L3",
                    therapy_type="CBT",
                    diagnosis_context=DiagnosisContext(primary="MDD"),
                    risk_context=RiskContext())
    assert ctx.l4_substage == "L4-L3"
    with pytest.raises(ValidationError):
        L5Context(l4_raw_output="你好", l4_substage="L4-L9",
                  therapy_type="CBT",
                  diagnosis_context=DiagnosisContext(),
                  risk_context=RiskContext())


def test_structured_summary_defaults():
    s = StructuredSummary(timestamp="2026-08-12T00:00:00Z",
                          summary="s",
                          structured_labels={},
                          evidence_refs=[EvidenceRef(source="a", finding="b", basis="c")],
                          confidence=0.85)
    assert s.layer_id == "L4"
    assert s.flags == []


def test_diagnostic_report_parses_sample_and_ignores_extra():
    raw = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    report = DiagnosticReport.model_validate(raw)
    assert report.diagnosis.primary.disorder == "MDD"
    assert report.diagnosis.primary.confidence == 0.87
    assert report.recommended_therapy.primary == "CBT"
    assert report.recommended_therapy.addon == ["CBT-I"]
    assert report.verifier_routing == "pass"
    anchor = report.to_anchor()
    assert anchor.therapy_type == "CBT"
    assert "CBT-I" in anchor.plugins
    assert "Insomnia Disorder" in anchor.comorbidities
    assert "Bipolar II" in anchor.excluded
