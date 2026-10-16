"""在 VS Code 中直接运行的 Patient Agent 测试入口。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from patient_agent.config import Settings
from patient_agent.evaluator import evaluate_scenario
from patient_agent.factory import build_library, build_stack
from patient_agent.generator import TurnContext
from patient_agent.llm import build_llm_client
from patient_agent.llm.base import LLMClient
from patient_agent.report import build_report, render_markdown, write_report
from patient_agent.reverse import (
    ReverseReport,
    build_report as build_reverse_report,
    load_reverse_cases,
)
from patient_agent.runner import RunOptions
from patient_agent.recorder import SessionRecorder, TurnRecord

ROOT = Path(__file__).resolve().parent

VERDICT_LABEL = {
    "clean": "干净：守卫放行，也没有病例外内容",
    "blocked_by_guard": "守卫拦截：候选没有进入系统（但模型可能确实编了，看守卫原因）",
    "inconclusive": "没得出结论：复核环节没跑完就被拦下（既没放行，也没有编造证据）",
    "escaped": "幻觉逃逸：守卫放行，但回答含病例外内容",
    "checker_ineffective": "检查器失效：对照幻觉样本没被守卫拦下",
}

USAGE = """用法：
  python run_patient_agent.py                              交互式菜单（1 手动 / 2 自动回放 / 3 反向检查）
  python run_patient_agent.py --reverse                    反向病例幻觉检查（离线，不发 API 请求）
  python run_patient_agent.py --reverse --api              反向病例幻觉检查（用 .env 里的真实模型）
  python run_patient_agent.py --reverse --reverse-id REV_DB_P0003
  python run_patient_agent.py --reverse --reverse-dir reverse_cases --out runs/my-reverse

说明：
  --reverse 走 reverse_cases/ 里的临时反向病例；它们不在正式病例库里，
  不参与 list / run-all / 验收测试，检查完可以直接删除整个目录。
  退出码：0 通过；1 有幻觉逃逸或检查器失效；2 用法或配置错误。
"""


class TestLLM(LLMClient):
    """测试时显式暴露模型错误，避免生成器静默退回参考台词。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = getattr(client, "name", "unknown")
        self.model = getattr(client, "model", self.name)

    def is_available(self) -> bool:
        return self.client.is_available()

    def complete(self, **kwargs) -> str:
        try:
            text = self.client.complete(**kwargs)
        except Exception as exc:
            # 不打印第三方异常正文，避免网关错误中包含密钥或请求内容。
            raise RuntimeError(f"模型请求失败：{type(exc).__name__}") from exc
        if not text.strip():
            raise RuntimeError("模型返回空文本，测试已停止。")
        return text


def choose(prompt: str, count: int, default: int = 1) -> int:
    while True:
        value = input(f"{prompt} [{default}]：").strip() or str(default)
        if value.isdigit() and 1 <= int(value) <= count:
            return int(value)
        print(f"请输入 1～{count}。")


def show_action(stack, action, *, question: str = "", recorder=None, record_path=None):
    action, result, retries = stack.runner.validate_action(action, TurnContext(last_reply=question))
    if recorder is not None:
        attempts = [a.model_copy(deep=True) for a in stack.runner.last_attempts]
        if result.ok:
            attempts[-1].submission_status = "displayed_only"
        recorder.turns.append(TurnRecord(turn_index=len(recorder.turns), action_kind=action.kind,
            submitted=False, submission_status="displayed_only" if result.ok else "blocked",
            final_text=action.text if result.ok else "", rejected_text=action.text if not result.ok else "",
            patient_fact_id=action.fact_id, patient_denial=action.is_denial,
            main_layer="", attempts=attempts, guard_ok=result.ok, retries=retries,
            violations=[v.model_dump(mode="json") for v in result.violations]))
        if record_path:
            recorder.dump(record_path)
    print(f"\n{'患者' if result.ok else '被拒绝的候选（未提交）'}：{action.text}")
    print(f"[调试] 事实编号：{action.fact_id or '未选择事实'}；原因：{action.reason}")
    print(f"[守卫] {'通过' if result.ok else '未通过'}")
    for violation in result.violations:
        print(f"  {violation.code.value}：{violation.message}")
    print("", flush=True)
    return action, result.ok


def manual_test(stack) -> int:
    print("\n你扮演医生，Patient Agent 扮演所选病例中的患者。")
    print("当前在模拟 L2 普通访谈；不连接真实诊疗后端。")
    print("/exit 结束；/facts 查看病例事实；/prompt 查看上一轮提示词。")
    print("守卫通过只是现有检查结果，仍需对照病例核查是否编造。")
    recorder = SessionRecorder(scenario_id=stack.scenario.scenario_id,
        scenario_version=stack.scenario.version, scenario_title=stack.scenario.title,
        llm_mode=stack.runner.llm_mode, llm_name=stack.runner.llm_name, system_api_name="manual_doctor")
    record_path = ROOT / "runs" / ("manual-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]) / "recorder.json"
    print(f"本次证据链记录：{record_path}")
    last_action = stack.generator.opening()
    last_action, accepted = show_action(stack, last_action, recorder=recorder, record_path=record_path)
    if not accepted:
        return 1
    history: list[tuple[str, str]] = []
    turn = 0
    while True:
        question = input("医生：").strip()
        if question == "/exit":
            return 0
        if question == "/facts":
            for fact in stack.scenario.revealable_facts():
                print(f"{fact.fact_id}：{fact.denial_utterance if fact.denial_expected else fact.utterance}")
            continue
        if question == "/prompt":
            print(last_action.prompt_system)
            print(last_action.prompt_user)
            continue
        if not question:
            continue
        turn += 1
        # 保持医生提问→患者作答的顺序，并让最后一句提问进入现有提示词。
        context = TurnContext(
            turn_index=turn,
            last_reply=question,
            recent_turns=history[-3:] + [(question, "")],
            waiting_type="free_text",
            main_layer="L2",
        )
        print("患者生成中……", flush=True)
        last_action = stack.generator.next_action(context)
        last_action, accepted = show_action(stack, last_action, question=question, recorder=recorder, record_path=record_path)
        if not accepted:
            print("守卫未放行，测试停止；候选回答不会进入对话历史。")
            return 1
        history.append((question, last_action.text))


def automatic_test(stack, settings: Settings) -> int:
    print("\n自动回放中：患者按病例脚本回应，医生端使用内置模拟系统。", flush=True)
    recorder = stack.runner.run()
    evaluation = evaluate_scenario(stack.scenario, recorder)
    for turn in recorder.turns:
        print(f"\n第 {turn.turn_index + 1} 轮 · {turn.main_layer} · {turn.elapsed_ms:.1f} ms")
        print(f"患者：{turn.patient_text}" if turn.submitted else f"被拒绝的候选（未提交）：{turn.rejected_text}")
        print(f"系统：{turn.system_reply}")
    report = build_report([evaluation])
    print(render_markdown(report))
    out = ROOT / "runs" / ("interactive-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    recorder.dump(out / "recorder.json")
    write_report(report, out)
    print(f"完整记录及报告：{out}")
    return 0 if evaluation.ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# 反向病例幻觉检查（临时工具）
# ─────────────────────────────────────────────────────────────────────────────


def render_reverse_report(report: ReverseReport) -> None:
    """把反向检查结果按「人怎么核对」的顺序打到屏幕上。"""

    tally = report.counts()
    print(f"\n反向病例：{len(report.cases)} 个；探针：{tally['probes']} 个；模型：{report.llm_mode}（{report.llm_name}）")
    for case in report.cases:
        counts = case.counts()
        print(f"\n{'=' * 78}")
        print(f"{case.reverse_id} · {case.title}")
        print(f"源病例：{case.source_scenario} v{case.source_version}")
        print(f"为什么要反着问：{case.why}")
        print(f"本病例：clean {counts['clean']} / blocked_by_guard {counts['blocked_by_guard']} / "
              f"inconclusive {counts['inconclusive']} / escaped {counts['escaped']} / "
              f"checker_ineffective {counts['checker_ineffective']}")
        for probe in case.probes:
            print(f"\n--- {probe.probe_id}（{probe.kind}）→ {VERDICT_LABEL[probe.verdict]}")
            print(f"  医生（诱导）：{probe.doctor}")
            if probe.trap:
                print(f"  陷阱：{probe.trap}")
            print(f"  病例本该用的依据：{probe.expect_fact_id or '（病例不该作答）'}；本轮实际选用：{probe.selected_fact_id or '（未选事实）'}")
            print(f"  患者回答：{probe.patient_reply or '（空）'}")
            guard_line = f"  守卫：{'放行' if probe.guard_ok else '拦截'}"
            guard_line += f"（{', '.join(probe.guard_codes) or '无违规'}；复核模式 {probe.review_mode}"
            guard_line += f"{'；判定 ' + '/'.join(probe.review_statuses) if probe.review_statuses else ''}；重说 {probe.retries} 次）"
            print(guard_line)
            for message in probe.guard_messages:
                print(f"    · {message}")
            if probe.findings:
                for finding in probe.findings:
                    tag = "阻断" if finding.blocking else "提示"
                    print(f"  独立检测[{tag}] {finding.code}：{finding.message}"
                          f"（证据：{', '.join(finding.evidence) or '—'}）")
            else:
                print("  独立检测：未发现病例外内容")
            if probe.fabrication_attempts:
                print(f"  重说前被拦下的候选（{len(probe.fabrication_attempts)} 条，共重说 {probe.retries} 次）：")
                for snippet in probe.fabrication_snippets():
                    print(f"    · {snippet}")
            print(f"  对照注入：{'已拦下' if probe.control_blocked else '未被拦下！'}"
                  f"（{', '.join(probe.control_codes) or '无违规'}）：{probe.control_sample}")
    print(f"\n{'=' * 78}")
    print("汇总")
    print(f"  探针 {tally['probes']} 个：clean {tally['clean']}；blocked_by_guard {tally['blocked_by_guard']}；"
          f"inconclusive {tally['inconclusive']}；escaped {tally['escaped']}；"
          f"checker_ineffective {tally['checker_ineffective']}")
    print(f"  模型疑似编造（被拦 + 逃逸，不含复核失效）：{tally['fabrication_suspected']} 个探针；"
          f"其中重说前被拦下的加戏候选 {tally['fabrication_attempts']} 条")
    print(f"  守卫违规码分布：{report.guard_code_tally() or '无'}")
    print(f"  结论：{report.conclusion()}"
          + ("" if report.ok() else "——看上面 escaped / checker_ineffective / inconclusive 的探针"))


def reverse_test(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    reverse_ids: list[str] | None = None,
    reverse_dir: Path | None = None,
    out_dir: Path | None = None,
) -> int:
    """用临时反向病例检查患者 Agent 有没有编造病例外事实。"""

    probe_root = reverse_dir or (ROOT / "reverse_cases")
    cases = load_reverse_cases(probe_root)
    if reverse_ids:
        wanted = set(reverse_ids)
        cases = [case for case in cases if case.reverse_id in wanted]
        missing = wanted - {case.reverse_id for case in cases}
        if missing:
            print(f"反向病例索引里没有：{', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    if not cases:
        print("没有可跑的反向病例。", file=sys.stderr)
        return 2

    library = build_library(settings)
    pairs = []
    for case in cases:
        scenario = library.load(case.source_scenario, case.source_version or None)
        pairs.append((case, scenario))

    if settings.llm_mode != "api":
        print("\n提醒：当前不是真实模型模式。离线模式下患者台词就是病例参考台词，"
              "探针只会验证「检查器本身有效」（对照注入必须被拦下）；")
        print("      要看模型会不会编造，请用真实 LLM（菜单选 1，或 --reverse --api）。")

    report = build_reverse_report(pairs, settings, llm=llm, reverse_dir=probe_root)
    render_reverse_report(report)

    out = out_dir or (ROOT / settings.out_dir / ("reverse-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]))
    json_path, markdown_path = report.dump(out)
    print(f"\n机器可读：{json_path}")
    print(f"人可读汇总：{markdown_path}")
    return 0 if report.ok() else 1


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────


class Options:
    """极简命令行参数（不引 argparse，和本文件的交互式风格保持一致）。"""

    def __init__(self) -> None:
        self.reverse = False
        self.api = False
        self.help = False
        self.reverse_dir: Path | None = None
        self.out_dir: Path | None = None
        self.reverse_ids: list[str] = []


def parse_args(argv: list[str]) -> Options:
    options = Options()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("-h", "--help"):
            options.help = True
        elif token == "--reverse":
            options.reverse = True
        elif token == "--api":
            options.api = True
        elif token == "--offline":
            options.api = False
        elif token == "--reverse-dir":
            index += 1
            options.reverse_dir = Path(argv[index] if index < len(argv) else "")
        elif token == "--reverse-id":
            index += 1
            options.reverse_ids.append(argv[index] if index < len(argv) else "")
        elif token == "--out":
            index += 1
            options.out_dir = Path(argv[index] if index < len(argv) else "")
        else:
            raise ValueError(f"未知参数：{token}")
        index += 1
    if options.reverse_dir is not None and not str(options.reverse_dir):
        raise ValueError("--reverse-dir 需要给一个目录")
    return options


def build_settings(llm_mode: str) -> Settings:
    settings = Settings.from_env(ROOT / ".env", llm_mode=llm_mode, system_api_mode="mock")
    scenario_path = Path(settings.scenario_dir)
    settings.scenario_dir = str(scenario_path if scenario_path.is_absolute() else ROOT / scenario_path)
    settings.validate()
    return settings


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="replace")

    options = parse_args(sys.argv[1:])
    if options.help:
        print(USAGE)
        return 0
    if options.reverse:
        settings = build_settings("api" if options.api else "offline")
        if options.out_dir is not None and not options.out_dir.is_absolute():
            options.out_dir = ROOT / options.out_dir
        client = TestLLM(build_llm_client(settings))
        return reverse_test(settings, llm=client, reverse_ids=options.reverse_ids or None,
                            reverse_dir=options.reverse_dir, out_dir=options.out_dir)

    print("Patient Agent · 可直接启动的测试")
    print(f"API 配置文件：{ROOT / '.env'}")
    print("1. 真实 LLM（使用 .env 配置，请求会发送至模型服务并可能产生费用）")
    print("2. 离线模式（病例参考台词，无 API 请求）")
    mode = choose("选择模型模式", 2)
    settings = build_settings("api" if mode == 1 else "offline")

    print("\n1. 手动提问（检查关键词、同义问法和事实一致性）")
    print("2. 自动完整病例回放（结束后显示完整对话并生成报告）")
    print("3. 反向病例幻觉检查（医生侧诱导提问 + 独立检测 + 对照注入）")
    test_mode = choose("选择测试方式", 3)
    if mode == 1:
        print(f"\n模型：{settings.openai_model}；地址：{settings.openai_base_url}")
    client = TestLLM(build_llm_client(settings))
    if test_mode == 3:
        return reverse_test(settings, llm=client)

    library = build_library(settings)
    entries = library.entries()
    if not entries:
        raise ValueError("病例库为空。")
    print(f"\n病例库：{settings.scenario_dir}")
    for number, entry in enumerate(entries, start=1):
        print(f"{number}. {entry.scenario_id} — {entry.title}")
    selected = entries[choose("选择病例", len(entries)) - 1]
    scenario = library.load(selected.scenario_id)
    stack = build_stack(scenario, settings, llm=client, options=RunOptions(max_retries=settings.max_generation_retries))
    return manual_test(stack) if test_mode == 1 else automatic_test(stack, settings)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n测试已结束。")
    except Exception as error:
        print(f"\n测试停止：{type(error).__name__}：{error}", file=sys.stderr)
        raise SystemExit(2)
