"""L3 诊断推理层演示。用法: python examples/demo_l3.py"""

import json

from layers.layer3.diagnosis.clinicr import CLINICR
from layers.layer3.therapy.therapy_selector import TherapySelector
from layers.layer3.instructions.instruction_generator import InstructionGenerator
from layers.layer3.api import build_report


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    patient_profile = {
        "PHQ9": 16,
        "GAD7": 9,
        "MDQ": 3,
        "PCL5": 18,
        "ISI": 14,
        "Y_BOCS": 0,
        "ADNM": 0,
    }
    safety_context = {
        "risk_level": 1,
        "sensitive_topics": ["trauma"],
        "l1_flags": ["monitor_suicidal_ideation", "avoid_trauma_probing"],
    }

    print_section("CLINICR 3-Step Differential Diagnosis")
    clinicr = CLINICR()
    result = clinicr.run(patient_profile, safety_context=safety_context)

    print("\n[Step 1] Generate Candidates (MCQ-CLINICR)")
    for i, c in enumerate(result["candidates"], 1):
        print(f"  Candidate {i}: {c['label']} (Confidence: {c['confidence']})")

    print("\n[Step 2] DSM-5 Elimination (ELIMINATIVE)")
    for c in result["checked"]:
        status_icon = "PASS" if c["status"] == "supported" else "FAIL"
        print(f"  [{status_icon}] {c['label']} - {c['status']}")

    print("\n[Step 3] Reward Model Verifier")
    primary = result["diagnosis"]["primary"]
    if primary:
        print(f"  FINAL: {primary['disorder']}  confidence={primary['confidence']}")
        print(f"  DSM-5 criteria met: {primary['dsm5_criteria_met']}")
        print(f"  Verifier routing: {result['verifier_routing']}")
    else:
        print("  No clear diagnosis")
        return

    print_section("Therapy Selection")
    selector = TherapySelector()
    therapy = selector.run(primary["disorder"], patient_profile)
    print(f"  Basic: {therapy['basic_therapy']}  Plugins: {therapy['plugins']}")
    print(f"  Contraindications: {therapy['contraindications']}")

    print_section("DiagnosticReport (L3 -> L4)")
    report = build_report(patient_profile, result, therapy, safety_context)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print_section("L1-L4 Treatment Instructions")
    generator = InstructionGenerator()
    instructions = generator.generate(result["diagnosis"], therapy, patient_profile)
    for key in ("L1", "L2", "L3", "L4"):
        print(f"\n[{key}] {instructions[key]}")


if __name__ == "__main__":
    main()
