"""命令行入口：列病例、跑单个病例、批量回放并出报告。

用法（默认离线确定性模式，不需要任何 API key）::

    python -m patient_agent list
    python -m patient_agent run S01_low_risk_depression
    python -m patient_agent run-all --out runs/latest

想连真实模型或真实编排器时再显式打开::

    python -m patient_agent run-all --llm-mode auto
    python -m patient_agent run-all --system-api http --system-api-base-url http://127.0.0.1:8000

退出码：0 全部通过；1 有病例未通过评测；2 用法或配置错误。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from patient_agent.config import Settings
from patient_agent.errors import PatientAgentError
from patient_agent.factory import (
    build_library,
    run_all,
    run_and_evaluate,
)
from patient_agent.report import render_markdown, write_report
from patient_agent.runner import RunOptions

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文与箭头会乱码；能改就改成 UTF-8。"""

    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 极少见
            pass


def _build_settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.scenario_dir:
        overrides["scenario_dir"] = args.scenario_dir
    if args.out:
        overrides["out_dir"] = args.out
    if args.llm_mode:
        overrides["llm_mode"] = args.llm_mode
    if args.system_api:
        overrides["system_api_mode"] = args.system_api
    if args.system_api_base_url:
        overrides["system_api_base_url"] = args.system_api_base_url
    settings = Settings.from_env(args.env_file, **overrides)
    settings.validate()
    return settings


def _print_transcript(recorder) -> None:
    for turn in recorder.turns:
        head = f"[{turn.turn_index:>2}] {turn.main_layer}/{turn.action_kind}"
        if turn.secondary_layers:
            head += f" (+{'/'.join(turn.secondary_layers)})"
        head += f" -> {turn.waiting_type}"
        print(head)
        if turn.patient_text:
            print(f"     患者> {turn.patient_text}")
        if not turn.submitted:
            print(f"     被拒绝（未提交）> {turn.rejected_text}")
        if turn.system_reply:
            print(f"     系统< {turn.system_reply}")
        if turn.note:
            print(f"     备注：{turn.note}")


def _print_evaluation(evaluation) -> None:
    status = "通过" if evaluation.ok else "不通过"
    print("")
    print(
        f"评测：{status} | 轮数 {evaluation.turn_count} "
        f"(治疗 {evaluation.treatment_turns}) | 终止 {evaluation.termination_observed or '未终止'}"
    )
    print(f"覆盖层：{'、'.join(evaluation.layers_observed) or '无'}")
    print(f"停止原因：{evaluation.stop_reason}")
    for check in evaluation.checks:
        mark = "OK " if check.ok else "FAIL"
        print(f"  [{mark}] {check.name}：{check.detail}")


# ---------------------------------------------------------------------- 子命令
def _cmd_list(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    library = build_library(settings)
    entries = library.entries()
    print(f"病例库：{Path(settings.scenario_dir).resolve()}（共 {len(entries)} 例）")
    for entry in entries:
        versions = "、".join(entry.versions)
        print(f"  - {entry.scenario_id}  v{entry.latest}  [{versions}]  {entry.title}")
        if entry.tags:
            print(f"      标签：{'、'.join(entry.tags)}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    library = build_library(settings)
    scenario = library.load(args.scenario_id, args.version)
    options = RunOptions(max_retries=settings.max_generation_retries)
    recorder, evaluation = run_and_evaluate(scenario, settings, options=options)

    print(f"病例：{scenario.scenario_id} v{scenario.version} — {scenario.title}")
    print(f"会话：{recorder.session_id} | 模型：{recorder.llm_mode}/{recorder.llm_name}")
    print("-" * 72)
    _print_transcript(recorder)
    _print_evaluation(evaluation)

    out_dir = Path(settings.out_dir) / scenario.scenario_id
    record_path = recorder.dump(out_dir / "recorder.json")
    print("")
    print(f"逐轮记录已写入：{record_path}")
    return EXIT_OK if evaluation.ok else EXIT_FAILED


def _cmd_run_all(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    only = [item.strip() for item in (args.only or "").split(",") if item.strip()] or None
    options = RunOptions(max_retries=settings.max_generation_retries)

    outcome = run_all(settings, scenario_ids=only, options=options)

    out_dir = Path(settings.out_dir)
    for recorder in outcome.records:
        recorder.dump(out_dir / recorder.scenario_id / "recorder.json")
    json_path, md_path = write_report(outcome.report, out_dir)

    print(render_markdown(outcome.report), end="")
    print(f"报告已写入：{json_path}")
    print(f"           {md_path}")

    if args.verbose:
        for evaluation in outcome.evaluations:
            print("")
            print("=" * 72)
            print(f"病例：{evaluation.scenario_id}")
            _print_evaluation(evaluation)

    return EXIT_OK if outcome.report.failed == 0 else EXIT_FAILED


# ---------------------------------------------------------------------- 参数
def _add_common_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """给主解析器和各子解析器挂同一批选项。

    子解析器那份用 ``SUPPRESS`` 当默认值：只有用户真的写在子命令后面时才会赋值，
    否则不会把主解析器已经解析到的值覆盖成 None。这样 ``--out`` 放在子命令前后都行。
    """

    def default(value: object) -> object:
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--scenario-dir",
        default=default(None),
        help="病例库根目录（默认 scenarios）",
    )
    parser.add_argument("--out", default=default(None), help="输出目录（默认 runs）")
    parser.add_argument(
        "--llm-mode",
        choices=["offline", "auto", "api"],
        default=default(None),
        help="offline 规则渲染（默认）/ auto 有 key 就用真模型 / api 强制真模型",
    )
    parser.add_argument(
        "--system-api",
        choices=["mock", "http"],
        default=default(None),
        help="mock 内置五层桩（默认）/ http 连真实编排器",
    )
    parser.add_argument(
        "--system-api-base-url", default=default(None), help="真实编排器地址"
    )
    parser.add_argument("--env-file", default=default(None), help="指定 .env 路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patient-agent",
        description="心理诊疗系统 Patient Agent：病例回放与自动评测",
    )
    _add_common_options(parser, suppress_defaults=False)

    sub = parser.add_subparsers(dest="command", required=True)

    # 子解析器各自再挂一份公共选项，允许「子命令在前、选项在后」的写法。
    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text)
        _add_common_options(child, suppress_defaults=True)
        return child

    p_list = add("list", "列出病例库")
    p_list.set_defaults(func=_cmd_list)

    p_run = add("run", "跑单个病例")
    p_run.add_argument("scenario_id", help="病例 ID，例如 S01_low_risk_depression")
    p_run.add_argument("--version", default=None, help="病例版本，默认取 latest")
    p_run.set_defaults(func=_cmd_run)

    p_all = add("run-all", "批量回放并生成报告")
    p_all.add_argument("--only", default=None, help="逗号分隔的病例 ID 子集")
    p_all.add_argument("--verbose", action="store_true", help="额外打印每个病例的评测明细")
    p_all.set_defaults(func=_cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"找不到文件：{exc}", file=sys.stderr)
        return EXIT_USAGE
    except PatientAgentError as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"配置有问题：{exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
