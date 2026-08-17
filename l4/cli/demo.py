"""L4 治疗对话层 CLI 演示。

用法：
    python -m l4.cli.demo            # 交互式（需 stdin 终端）
    python -m l4.cli.demo --scripted # 脚本化演示，自动走完 L4-L1 → L4-L4
默认使用 MockBackend + MockReviewer，无需 API 密钥。
"""

import argparse
import json
import sys
from pathlib import Path

from l4.config import Settings
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.service import L4Service, SessionRequest, TurnRequest

ROLE = RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化")
BOUNDARY = BoundaryInstruction(
    absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来"],
    safety_trigger="用户表达自杀意图→立即触发安全协议",
)
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
    path = Path(__file__).resolve().parents[2] / "examples" / "sample_diagnostic_report.json"
    return DiagnosticReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _print_turn(service: L4Service, session_id: str, user_message: str,
                turn: TurnInstruction) -> None:
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
    service = L4Service(Settings(_env_file=None))
    session_id = service.create_session(SessionRequest(
        role=ROLE, report=_load_report(), boundary=BOUNDARY, safety=SAFETY))
    print("== L4 治疗对话层 脚本化演示（MockBackend + MockReviewer）==")
    print(f"诊断锚点: {service.store.get(session_id).anchor.primary}")
    for message, turn in SCRIPTED_TURNS:
        _print_turn(service, session_id, message, turn)
    print("\n演示结束。四阶段 L4-L1 → L4-L4 已依次推进。")


def run_interactive() -> None:
    service = L4Service(Settings(_env_file=None))
    session_id = service.create_session(SessionRequest(
        role=ROLE, report=_load_report(), boundary=BOUNDARY, safety=SAFETY))
    print("== L4 交互式演示：输入消息，Ctrl+C 退出 ==")
    stage_idx = 0
    while True:
        try:
            message = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not message:
            continue
        force = SCRIPTED_TURNS[stage_idx][1].force_substage if stage_idx < len(SCRIPTED_TURNS) else None
        stage_idx = min(stage_idx + 1, len(SCRIPTED_TURNS) - 1)
        turn = TurnInstruction(
            goal="跟随患者推进治疗", technique="反映+开放式提问",
            forbidden=["建议", "确诊"],
            force_substage=force,
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
