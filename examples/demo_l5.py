"""L1 + L5 输出审核演示。用法: python examples/demo_l5.py [用户消息]"""

import json
import sys
from pathlib import Path

from layers.layer1.detector import safety_detect
from layers.layer5.reviewer import review_response
from layers.layer4.generation.summary_builder import build_l5_context
from shared.schemas.treatment_instructions import SafetyPassport, TreatmentInstructions, TurnInstruction


def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else input("User: ")

    safety_context = safety_detect(user_input)
    print("===== L1 SafetyPassport =====")
    print(json.dumps(safety_context, ensure_ascii=False, indent=2))

    if safety_context.get("crisis_flag"):
        print("\n[危机级] 终止常规流程，执行危机干预协议。")
        return

    l4_output = input("\n模拟L4回复: ")

    # 仅演示审核接口：使用仓库模拟诊断的指令，不是对本次输入的真实诊断。
    sample = json.loads(Path(__file__).with_name("sample_diagnostic_report.json").read_text(encoding="utf-8"))
    instructions = TreatmentInstructions.model_validate(sample["instructions"])
    instructions.L3 = TurnInstruction(
        goal="探索患者当前困扰", technique="共情反映", source_message=user_input,
        force_substage="L4-L1", forbidden=["确诊", "开药"],
    )
    contextual_safety_context = build_l5_context(
        l4_output, "L4-L1", SafetyPassport.model_validate(safety_context),
        instructions, user_input, [],
    ).model_dump(mode="json")

    result = review_response(contextual_safety_context)
    print("\n===== L5 ContextualSafetyContext =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
