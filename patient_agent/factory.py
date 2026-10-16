"""把各层装配成可运行的一条链：设置 → 客户端 + 生成器 + 守卫 → 运行器。

CLI 与测试都从这里取装配逻辑，避免「两处参数不一样导致跑出来的东西不同」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from patient_agent.client import (
    MockConfig,
    MockSystemApiClient,
    SystemApiClient,
    load_http_client,
)
from patient_agent.config import Settings
from patient_agent.evaluator import ScenarioEvaluation, evaluate_scenario
from patient_agent.generator import PatientGenerator
from patient_agent.guard import ConsistencyGuard
from patient_agent.llm import LLMClient, build_llm_client
from patient_agent.library import ScenarioLibrary
from patient_agent.recorder import SessionRecorder
from patient_agent.report import BatchReport, build_report
from patient_agent.runner import RunOptions, SessionRunner
from patient_agent.schemas import PatientScenario
from patient_agent.semantic_guard import SemanticFactVerifier


@dataclass
class SessionStack:
    """一次会话所需的全部部件。"""

    scenario: PatientScenario
    client: SystemApiClient
    llm: LLMClient
    generator: PatientGenerator
    guard: ConsistencyGuard
    runner: SessionRunner


def build_library(settings: Settings) -> ScenarioLibrary:
    return ScenarioLibrary(settings.scenario_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 5】装配处：把「选哪个客户端」落到一行分支
# 接入什么：真实编排器（http）或内置五层桩（mock）。
# 怎么打开：settings.system_api_mode / system_api_base_url / system_api_token
#           （来自环境变量，见 config.py【接入点 9】）。
# 将来若有第三种后端（gRPC、进程内直连编排器），在这里加一个分支即可，
# 运行器/守卫/评测器都不需要知道对面是什么。
# ─────────────────────────────────────────────────────────────────────────────


def build_client(settings: Settings) -> SystemApiClient:
    """按 ``system_api_mode`` 造客户端：默认内置 Mock，http 走真实编排器。"""

    if settings.system_api_mode == "http":
        return load_http_client(
            settings.system_api_base_url,
            timeout=settings.system_api_timeout,
            token=settings.system_api_token,
        )
    return MockSystemApiClient(MockConfig(latency_ms=0.0))


def build_stack(
    scenario: PatientScenario,
    settings: Settings,
    *,
    client: SystemApiClient | None = None,
    llm: LLMClient | None = None,
    options: RunOptions | None = None,
) -> SessionStack:
    """装配一个病例的运行栈（客户端/模型可注入，便于测试替换）。"""

    resolved_client = client or build_client(settings)
    resolved_llm = llm or build_llm_client(settings)
    generator = PatientGenerator(scenario, resolved_llm, structured_output=settings.use_real_llm)
    guard = ConsistencyGuard(scenario, verifier=SemanticFactVerifier(resolved_llm) if settings.use_real_llm else None)
    runner = SessionRunner(
        scenario,
        resolved_client,
        generator,
        guard,
        options=options or RunOptions(),
        llm_mode=settings.llm_mode,
        llm_name=getattr(resolved_llm, "model", None) or getattr(resolved_llm, "name", resolved_llm.__class__.__name__),
    )
    return SessionStack(
        scenario=scenario,
        client=resolved_client,
        llm=resolved_llm,
        generator=generator,
        guard=guard,
        runner=runner,
    )


def run_scenario(
    scenario: PatientScenario,
    settings: Settings,
    *,
    client: SystemApiClient | None = None,
    llm: LLMClient | None = None,
    options: RunOptions | None = None,
) -> SessionRecorder:
    stack = build_stack(scenario, settings, client=client, llm=llm, options=options)
    return stack.runner.run()


def run_and_evaluate(
    scenario: PatientScenario,
    settings: Settings,
    *,
    client: SystemApiClient | None = None,
    llm: LLMClient | None = None,
    options: RunOptions | None = None,
) -> tuple[SessionRecorder, ScenarioEvaluation]:
    recorder = run_scenario(scenario, settings, client=client, llm=llm, options=options)
    return recorder, evaluate_scenario(scenario, recorder)


@dataclass
class BatchOutcome:
    """批量回放的结果：逐病例记录 + 评测 + 汇总报告。"""

    settings: Settings
    records: list[SessionRecorder]
    evaluations: list[ScenarioEvaluation]
    report: BatchReport


def run_all(
    settings: Settings,
    *,
    scenario_ids: list[str] | None = None,
    version: str | None = None,
    options: RunOptions | None = None,
    client: SystemApiClient | None = None,
    llm: LLMClient | None = None,
    generated_at: str | None = None,
) -> BatchOutcome:
    """把病例库里（或指定子集）的病例全部跑一遍并评测。"""

    library = build_library(settings)
    ids = scenario_ids or library.ids()
    resolved_client = client or build_client(settings)
    resolved_llm = llm or build_llm_client(settings)

    records: list[SessionRecorder] = []
    evaluations: list[ScenarioEvaluation] = []
    for scenario_id in ids:
        scenario = library.load(scenario_id, version)
        recorder = run_scenario(
            scenario,
            settings,
            client=resolved_client,
            llm=resolved_llm,
            options=options,
        )
        records.append(recorder)
        evaluations.append(evaluate_scenario(scenario, recorder))

    report = build_report(evaluations, generated_at=generated_at)
    return BatchOutcome(
        settings=settings,
        records=records,
        evaluations=evaluations,
        report=report,
    )


def default_out_dir(settings: Settings, name: str = "latest") -> Path:
    return Path(settings.out_dir) / name
