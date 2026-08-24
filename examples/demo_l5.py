"""L1 + L5 输出审核演示。用法: python examples/demo_l5.py [用户消息]"""

import json
import sys

from layers.layer1.detector import safety_detect
from layers.layer5.reviewer import review_response


def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else input("User: ")

    safety_context = safety_detect(user_input)
    print("===== L1 SafetyPassport =====")
    print(json.dumps(safety_context, ensure_ascii=False, indent=2))

    if safety_context.get("crisis_flag"):
        print("\n[危机级] 终止常规流程，执行危机干预协议。")
        return

    l4_output = input("\n模拟L4回复: ")

    contextual_safety_context = {
        "l4_raw_output": l4_output,
        "l4_substage": "L4-L3",
        "therapy_type": "CBT+BA",
        "diagnosis_context": {
            "primary": "MDD (moderate, confidence=0.75)",
            "secondary": [],
            "suicidal_ideation": "passive_only" if "monitor_suicidal_ideation" in safety_context.get("flags", []) else "none",
        },
        "risk_context": {
            "risk_level": safety_context.get("risk_level", 0),
            "sensitive_topics": safety_context.get("sensitive_topics", []),
        },
    }

    result = review_response(contextual_safety_context)
    print("\n===== L5 ContextualSafetyContext =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
