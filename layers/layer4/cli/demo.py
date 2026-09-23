"""L4 治疗对话层 CLI 演示。

用法：
    python -m layers.layer4.cli.demo            # 交互式（需 stdin 终端）
    python -m layers.layer4.cli.demo --scripted # 脚本化演示，自动走完 L4-L1 → L4-L4
脚本演示使用受控指令与 MockBackend；交互模式使用 L3 指令 API 和 L4 生产配置。
"""

import argparse
import json
import sys
from pathlib import Path

from layers.layer4.config import Settings
from layers.layer4.schemas.instructions import (
    SafetyPassport,
    TurnInstruction,
)
from layers.layer4.schemas.l3_report import DiagnosticReport
from layers.layer4.service import L4Service, SessionRequest, TurnRequest
from layers.layer3.instructions.turn_planner import TurnInstructionPlanner

SAFETY = SafetyPassport(risk_level=1, sensitive_topics=[], flags=["monitor_suicidal_ideation"])

SCRIPTED_TURNS = [
    (
        "我最近工作压力很大，晚上总是睡不好。",
        TurnInstruction(goal="建立共情与治疗联盟，确认当前状态",
                        technique="开放式提问+反映",
                        forbidden=["建议", "确诊"],
                        force_substage="L4-L1"),
    ),
    (
        "是啊，我也不知道怎么改善。",
        TurnInstruction(goal="温和反馈初步印象，征求患者确认",
                        technique="汇总反馈+开放式确认",
                        forbidden=["确诊", "肯定是"],
                        force_substage="L4-L2"),
    ),
    (
        "我确实总是往坏处想，一收到工作消息就心跳加速。",
        TurnInstruction(goal="识别并挑战负性自动化思维",
                        technique="苏格拉底式提问",
                        forbidden=["建议", "确诊"],
                        plugin_guidance="核心议题中引导患者识别自动化思维，注意这只是念头不是事实",
                        force_substage="L4-L3"),
    ),
    (
        "好，我觉得可以试试。",
        TurnInstruction(goal="布置家庭作业并总结本轮",
                        technique="行为激活作业+睡眠日记",
                        forbidden=["建议", "确诊"],
                        force_substage="L4-L4"),
    ),
]


def _load_report() -> DiagnosticReport:
    path = Path(__file__).resolve().parents[3] / "examples" / "sample_diagnostic_report.json"
    return DiagnosticReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _print_turn(service: L4Service, session_id: str, user_message: str,
                turn: TurnInstruction) -> None:
    if not turn.turn_number:
        # 仅脚本演示：由显式脚本提供 L3 指令，不声称进行了真实模型评估。
        sd = service.store.get(session_id)
        code = {"L4-L1": "reflection", "L4-L2": "feedback", "L4-L3": "thought_identification", "L4-L4": "review"}[turn.force_substage]
        number = len(sd.l3_history) + 1
        turn_values = turn.model_dump()
        turn_values.update({
            "session_id": session_id, "turn_number": number, "source_message": user_message,
            "technique_code": code, "technique": sd.anchor.allowed_techniques[code],
            "forbidden": list(dict.fromkeys(sd.boundary.absolute_bans + turn.forbidden)),
            "plugin_guidance": None,
            "link_previous": sd.history[-2].content if sd.history else "",
            "rationale": "受控演示脚本给定的阶段和目标",
            "previous_goal_met": number > 1, "patient_ready": number > 1,
            "evidence": [{"source_id": f"turn:{number}:user", "quote": user_message}],
        })
        turn = TurnInstruction.model_validate(turn_values)
    out = service.handle_turn(TurnRequest(
        session_id=session_id, user_message=user_message, turn=turn))
    print(f"\n用户：{user_message}")
    print(f"[子阶段] {out.l4_substage}    [兜底] {out.fallback_used}")
    print(f"治疗师：{out.reply}")
    for i, r in enumerate(out.review_history, 1):
        print(f"  审核#{i}: harm={r.harm} boundary={r.boundary} quality={r.quality}"
              f"  reason={r.fail_reason or '-'}")
    if out.summary.flags:
        print(f"  flags: {out.summary.flags}")


def run_scripted() -> None:
    service = L4Service(Settings(_env_file=None, llm_provider="mock", review_mode="mock"))
    session_id = service.create_session(SessionRequest(
        report=_load_report(), safety=SAFETY))
    print("== L4 治疗对话层 脚本化演示（MockBackend + MockReviewer）==")
    print(f"诊断锚点: {service.store.get(session_id).anchor.primary}")
    for message, turn in SCRIPTED_TURNS:
        _print_turn(service, session_id, message, turn)
    print("\n演示结束。四阶段 L4-L1 → L4-L4 已依次推进。")


def run_interactive() -> None:
    service = L4Service()
    session_id = service.create_session(SessionRequest(
        report=_load_report(), safety=SAFETY))
    print("== L3 指令 → L4 API 集成演示（测试画像，非完整 L1 安全入口）：Ctrl+C 退出 ==")
    planner = TurnInstructionPlanner()
    while True:
        try:
            message = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not message:
            continue
        sd = service.store.get(session_id)
        turn = planner.plan(
            session_id=session_id, instructions=sd.instructions(),
            records=sd.treatment_records(), user_message=message,
            safety=sd.safety.model_dump(mode="json"),
        )
        _print_turn(service, session_id, message, turn)


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 治疗对话层演示")
    parser.add_argument("--scripted", action="store_true", help="脚本化自动演示")
    args = parser.parse_args()
    if args.scripted or not sys.stdin.isatty():
        run_scripted()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
