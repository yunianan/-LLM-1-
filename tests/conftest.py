"""测试公共夹子：路径、设置、病例库、运行栈。

刻意不让测试依赖当前工作目录：``PROJECT_ROOT`` 由 ``__file__`` 推出来，
因此 ``pytest`` 从任何目录启动都能找到 ``scenarios/``。

所有用例默认走离线确定性模式（``llm_mode=offline`` + 内置 Mock 五层桩），
不需要任何 API key，也不出网。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from patient_agent.client.mock_system import MockConfig, MockSystemApiClient
from patient_agent.config import Settings
from patient_agent.factory import build_stack
from patient_agent.generator import PatientGenerator
from patient_agent.guard import ConsistencyGuard
from patient_agent.llm import RuleBasedLLM
from patient_agent.library import ScenarioLibrary
from patient_agent.runner import RunOptions, SessionRunner
from patient_agent.schemas import PatientScenario

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = PROJECT_ROOT / "scenarios"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def scenario_dir() -> Path:
    assert SCENARIO_DIR.is_dir(), f"找不到病例库目录：{SCENARIO_DIR}"
    return SCENARIO_DIR


@pytest.fixture()
def settings(tmp_path: Path, scenario_dir: Path) -> Settings:
    """离线设置，输出目录指向 pytest 的临时目录，避免污染仓库。"""

    instance = Settings.from_env(
        environ={},
        scenario_dir=str(scenario_dir),
        out_dir=str(tmp_path / "runs"),
        llm_mode="offline",
        system_api_mode="mock",
    )
    instance.validate()
    return instance


@pytest.fixture(scope="session")
def library(scenario_dir: Path) -> ScenarioLibrary:
    return ScenarioLibrary(scenario_dir)


@pytest.fixture()
def all_scenarios(library: ScenarioLibrary) -> dict[str, PatientScenario]:
    return {scenario_id: library.load(scenario_id) for scenario_id in library.ids()}


def make_scenario(library: ScenarioLibrary, scenario_id: str) -> PatientScenario:
    return library.load(scenario_id)


@pytest.fixture()
def generator_factory(library: ScenarioLibrary):
    """造一个「病例 + 生成器」，默认用规则模型。

    每个用例拿到的是病例的**深拷贝**：用例里为了验证分支会临时改病例（例如把量表
    意愿改成拒答），改到共享缓存上会污染同一 session 里的其它用例。
    """

    def _make(scenario_id: str, llm=None) -> tuple[PatientScenario, PatientGenerator]:
        scenario = library.load(scenario_id).model_copy(deep=True)
        return scenario, PatientGenerator(scenario, llm or RuleBasedLLM())

    return _make


@pytest.fixture()
def guard_factory(library: ScenarioLibrary):
    def _make(scenario_id: str) -> tuple[PatientScenario, ConsistencyGuard]:
        scenario = library.load(scenario_id).model_copy(deep=True)
        return scenario, ConsistencyGuard(scenario)

    return _make


@pytest.fixture()
def runner_factory(library: ScenarioLibrary):
    """按病例 ID 造一个「已装配好的」运行栈。"""

    def _make(
        scenario_id: str,
        *,
        mock_config: MockConfig | None = None,
        options: RunOptions | None = None,
    ):
        scenario = library.load(scenario_id)
        client = MockSystemApiClient(mock_config or MockConfig())
        return build_stack(
            scenario,
            Settings(llm_mode="offline", system_api_mode="mock"),
            client=client,
            llm=RuleBasedLLM(),
            options=options or RunOptions(),
        )

    return _make
