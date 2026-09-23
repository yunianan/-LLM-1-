from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .clinicr import CLINICR
from .therapy_selector import TherapySelector
from .instruction_generator import InstructionGenerator
from .output_validator import L3OutputValidator

app = FastAPI(title="Diagnosis Layer API")


class DiagnoseRequest(BaseModel):
    PHQ9: float = 0
    GAD7: float = 0
    MDQ: float = 0
    PCL5: float = 0
    ISI: float = 0
    Y_BOCS: float = 0
    ADNM: float = 0
    RRS: float = 0
    AAQ_II: float = 0
    PHQ15: float = 0
    relapse_count: int = 0
    rhythm_disturbance: bool = False
    exposure_avoidant: bool = False
    ritual_behaviors: bool = False
    chronic_months: int = 0
    interpersonal_conflicts: int = 0
    interpersonal_impairment: bool = False
    single_focus: bool = False
    psychotic_symptoms: bool = False
    substance_active: bool = False
    severe_dissociation: bool = False
    risk_level: int = 0
    sensitive_topics: List[str] = []
    l1_flags: List[str] = []
    PHQ9_items: dict[str, float] = Field(default_factory=dict)
    key_symptoms_extracted: list[dict[str, Any]] = Field(default_factory=list)
    symptom_duration_days: int | None = None
    functional_impairment: bool | None = None
    mania_or_hypomania_history: bool | None = None
    substance_or_medical_cause: bool | None = None
    differential_diagnoses_assessed: list[str] | None = None
    evidence_conflicts: list[dict[str, Any]] | None = None


class DiagnoseResponse(BaseModel):
    layer_id: Literal["L3"] = "L3"
    timestamp: str
    diagnosis: dict
    verifier_score: float
    candidate_confidence: float
    verifier_routing: str
    recommended_therapy: dict
    severity_level: int
    clinical_notes_for_L4: str
    summary: str
    structured_labels: dict
    evidence_refs: list
    confidence: float
    flags: list
    instruction_to_next: str
    instructions: dict
    missing_dimensions: list[dict[str, str]] = Field(default_factory=list)
    provisional_diagnosis: dict | None = None
    verifier_checks: list[dict[str, Any]] = Field(default_factory=list)
    verifier_fail_reasons: list[str] = Field(default_factory=list)
    l3_validation: dict[str, Any]


def build_blocked_report(clinicr_result):
    missing = clinicr_result.get("missing_dimensions", [])
    routing = clinicr_result["verifier_routing"]
    if routing == "return_to_l2":
        action = "collect_missing_dimensions"
        instruction = "返回L2补充 Verifier 指定的缺失信息"
    elif routing == "manual_review":
        action = "manual_review"
        instruction = "证据存在冲突，转人工复核，不进入L4"
    else:
        action = "stop_diagnostic_route"
        instruction = "终验不支持当前诊断，停止进入L4"
    return {
        "layer_id": "L3",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "diagnosis": clinicr_result["diagnosis"],
        "provisional_diagnosis": clinicr_result.get("provisional_diagnosis"),
        "verifier_score": clinicr_result["verifier_score"],
        "candidate_confidence": (
            clinicr_result.get("provisional_diagnosis") or {}
        ).get("confidence", 0.0),
        "verifier_routing": routing,
        "verifier_checks": clinicr_result.get("verifier_checks", []),
        "verifier_fail_reasons": clinicr_result.get("verifier_fail_reasons", []),
        "recommended_therapy": {"primary": "", "addon": [], "contraindications": []},
        "severity_level": clinicr_result["severity_level"],
        "clinical_notes_for_L4": "",
        "summary": clinicr_result["summary"],
        "structured_labels": {
            "therapy_routing": {"primary": "", "addon": [], "l4_substage_priority": []},
            "wording_constraint": "信息不足，不得形成诊断结论或进入治疗对话",
            "risk_monitoring": [],
        },
        "evidence_refs": clinicr_result["evidence_refs"],
        "confidence": 0.0,
        "flags": clinicr_result["flags"],
        "instruction_to_next": instruction,
        "instructions": {
            "L2": {
                "action": action,
                "missing_dimensions": missing,
            }
        },
        "missing_dimensions": missing,
    }


THERAPY_SUBSTAGE_PRIORITY = {
    "CBT": ["behavioral_activation", "cognitive_restructuring", "sleep_hygiene"],
    "CBT+BA": ["behavioral_activation", "cognitive_restructuring", "sleep_hygiene"],
    "CBT-TF": ["trauma_processing", "cognitive_restructuring"],
    "CBT-I": ["sleep_hygiene", "stimulus_control", "sleep_restriction"],
    "CBT+IPSRT": ["behavioral_activation", "interpersonal_rhythm"],
}

PLUGIN_SUBSTAGE = {
    "MBCT": "mindfulness",
    "ACT": "cognitive_defusion",
    "ERP": "exposure_response_prevention",
    "EMDR": "trauma_processing",
    "CBT-I": "sleep_hygiene",
    "IPSRT": "interpersonal_rhythm",
    "IPT": "interpersonal_work",
    "SFBT": "solution_focused",
}

GUIDELINE_SOURCE = {
    "CBT+BA": "APA_2019; NICE_NG222",
    "CBT": "APA_2019; NICE_NG222",
    "CBT-TF": "ISTSS_2019; NICE_NG116",
    "CBT-I": "AASM_2021; ACP_Qaseem_2016",
    "CBT+IPSRT": "ISBD_2020; NICE_NG185",
}


def build_report(profile, clinicr_result, therapy, safety_context=None):
    diagnosis = clinicr_result["diagnosis"]
    primary = diagnosis.get("primary")
    therapy_type = therapy.get("basic_therapy", "CBT")

    priority = list(THERAPY_SUBSTAGE_PRIORITY.get(therapy_type, ["behavioral_activation", "cognitive_restructuring"]))
    for plugin in therapy.get("plugins", []):
        extra = PLUGIN_SUBSTAGE.get(plugin)
        if extra and extra not in priority:
            priority.append(extra)

    safety = safety_context or {}
    sensitive_topics = safety.get("sensitive_topics", [])
    safety_flags = safety.get("l1_flags", []) or safety.get("flags", [])

    contraindications = list(therapy.get("contraindications", []))
    if "trauma" in sensitive_topics:
        contraindications.append("避免深度创伤暴露（per L1敏感话题标记）")

    risk_monitoring = []
    if "monitor_suicidal_ideation" in safety_flags or "passive_si_monitor" in safety_flags:
        risk_monitoring.append("passive_si_to_active_si_escalation")

    evidence_refs = list(clinicr_result["evidence_refs"])
    evidence_refs.append({
        "type": "guideline",
        "therapy": f"{therapy_type}_for_{primary['disorder']}" if primary else therapy_type,
        "source": GUIDELINE_SOURCE.get(therapy_type, "APA_2019"),
    })
    for plugin in therapy.get("plugins", []):
        evidence_refs.append({
            "type": "guideline",
            "therapy": f"{plugin}_plugin",
            "source": "临床指南（触发条件：量表阈值+患者特征）",
        })

    focus_notes = []
    for stage in priority:
        focus_notes.append(f"({priority.index(stage) + 1}){stage}")

    flags = list(clinicr_result.get("flags", []))
    for f in safety_flags:
        if f not in flags:
            flags.append(f)

    if primary is None:
        raise HTTPException(status_code=400, detail="Cannot determine diagnosis")

    clinical_notes = (
        f"L4治疗焦点按序推进：{'→'.join(priority)}。"
        f"主诊断{primary['disorder']}（置信度{primary['confidence']}，"
        f"严重度{primary.get('severity', 'unknown')}）。"
        f"需温和监测自杀意念变化，若从被动转为主动，立即升级至L1安全协议。"
    )

    instruction_to_next = (
        f"L4对话应聚焦{'→'.join(priority[:2])}，避免涉及创伤暴露练习。"
        "L4-L2需温和反馈诊断（'初步印象'措辞），征求患者确认。"
    )

    return {
        "layer_id": "L3",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "diagnosis": diagnosis,
        "verifier_score": clinicr_result["verifier_score"],
        "candidate_confidence": primary["confidence"] if primary else 0.0,
        "verifier_routing": clinicr_result["verifier_routing"],
        "verifier_checks": clinicr_result.get("verifier_checks", []),
        "verifier_fail_reasons": clinicr_result.get("verifier_fail_reasons", []),
        "recommended_therapy": {
            "primary": therapy_type,
            "addon": therapy.get("plugins", []),
            "contraindications": contraindications,
        },
        "severity_level": clinicr_result["severity_level"],
        "clinical_notes_for_L4": clinical_notes,
        "summary": clinicr_result["summary"],
        "structured_labels": {
            "therapy_routing": {
                "primary": therapy_type,
                "addon": therapy.get("plugins", []),
                "l4_substage_priority": priority,
            },
            "wording_constraint": "使用'初步印象'措辞，禁止'确诊'",
            "risk_monitoring": risk_monitoring,
        },
        "evidence_refs": evidence_refs,
        "confidence": clinicr_result["confidence"],
        "flags": flags,
        "instruction_to_next": instruction_to_next,
    }


def diagnose(profile, safety_context=None):
    clinicr = CLINICR()
    result = clinicr.run(profile, safety_context=safety_context)

    if result["verifier_routing"] not in {"pass", "conditional"}:
        report = build_blocked_report(result)
        report["l3_validation"] = L3OutputValidator().validate(report)
        return report

    primary = result["diagnosis"]["primary"]
    if primary is None:
        raise HTTPException(status_code=400, detail="Cannot determine diagnosis")

    selector = TherapySelector()
    therapy = selector.run(primary["disorder"], profile)

    report = build_report(profile, result, therapy, safety_context)

    report["l3_validation"] = L3OutputValidator().validate(report)
    if not report["l3_validation"]["passed"]:
        report["flags"] = list(dict.fromkeys(report["flags"] + ["l3_output_validation_failed"]))
        report["instruction_to_next"] = "L3输出校验失败，留在L3修正诊断或疗法，不进入L4"
        report["instructions"] = {
            "L3": {
                "action": "correct_output",
                "failed_checks": [
                    item for item in report["l3_validation"]["checks"]
                    if item["status"] == "fail"
                ],
            }
        }
        return report

    generator = InstructionGenerator()
    instructions = generator.generate(result["diagnosis"], therapy, profile)

    report["instructions"] = instructions
    return report


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(request: DiagnoseRequest):
    profile = {
        "PHQ9": request.PHQ9,
        "GAD7": request.GAD7,
        "MDQ": request.MDQ,
        "PCL5": request.PCL5,
        "ISI": request.ISI,
        "Y_BOCS": request.Y_BOCS,
        "ADNM": request.ADNM,
        "RRS": request.RRS,
        "AAQ_II": request.AAQ_II,
        "PHQ15": request.PHQ15,
        "relapse_count": request.relapse_count,
        "rhythm_disturbance": request.rhythm_disturbance,
        "exposure_avoidant": request.exposure_avoidant,
        "ritual_behaviors": request.ritual_behaviors,
        "chronic_months": request.chronic_months,
        "interpersonal_conflicts": request.interpersonal_conflicts,
        "interpersonal_impairment": request.interpersonal_impairment,
        "single_focus": request.single_focus,
        "psychotic_symptoms": request.psychotic_symptoms,
        "substance_active": request.substance_active,
        "severe_dissociation": request.severe_dissociation,
        "PHQ9_items": request.PHQ9_items,
        "key_symptoms_extracted": request.key_symptoms_extracted,
        "symptom_duration_days": request.symptom_duration_days,
        "functional_impairment": request.functional_impairment,
        "mania_or_hypomania_history": request.mania_or_hypomania_history,
        "substance_or_medical_cause": request.substance_or_medical_cause,
        "differential_diagnoses_assessed": request.differential_diagnoses_assessed,
        "evidence_conflicts": request.evidence_conflicts,
    }
    safety_context = {
        "risk_level": request.risk_level,
        "sensitive_topics": request.sensitive_topics,
        "l1_flags": request.l1_flags,
    }
    return DiagnoseResponse(**diagnose(profile, safety_context))


@app.get("/health")
def health():
    return {"status": "ok", "service": "Diagnosis Layer"}


@app.get("/")
def root():
    return {"message": "Diagnosis Layer API is running", "endpoints": ["/diagnose", "/health"]}
