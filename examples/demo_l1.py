"""L1 安全评估层演示。用法: python examples/demo_l1.py [可选: 直接给一段文本]"""

import json
import sys

from layers.layer1.detector import safety_detect


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else input("User: ")
    result = safety_detect(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("crisis_flag"):
        print("\n[危机级] 终止常规流程，执行危机干预协议。")


if __name__ == "__main__":
    main()
