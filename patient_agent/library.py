"""PA2 版本化病例库。

目录约定（病例库可以被单独打包、也可以用别的工具生成）：

    scenarios/
      index.json                 # 索引：有哪些病例、各有几个版本、最新是哪个
      S01_safety_crisis/
        1.0.0.json               # 一个版本一个文件，文件名就是版本号
      S02_bipolar_exclusion/
        1.0.0.json

为什么要把版本落到文件名上：同一份病例改一句台词就可能让「不迎合」验证结果变化，
测试结论必须能追溯到「当时用的是哪个版本」。索引里保留了历史版本，改病例时新增
文件而不是就地覆盖，这样旧版本的回归结果永远可复现。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patient_agent.errors import ScenarioLibraryError, ScenarioValidationError
from patient_agent.schemas import PatientScenario

INDEX_FILENAME = "index.json"


class ScenarioEntry(BaseModel):
    """索引里的一条记录。"""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    versions: list[str] = Field(min_length=1)
    latest: str = ""
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""

    def resolve_version(self, version: str | None) -> str:
        if version is None:
            return self.latest or self.versions[-1]
        if version not in self.versions:
            raise ScenarioLibraryError(
                f"病例 {self.scenario_id} 没有版本 {version}，可用：{', '.join(self.versions)}"
            )
        return version


class ScenarioIndex(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    scenarios: list[ScenarioEntry] = Field(default_factory=list)


class ScenarioLibrary:
    """病例库的门面：读取索引、解析单个病例、做 Schema 校验。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._index: ScenarioIndex | None = None
        self._cache: dict[tuple[str, str], PatientScenario] = {}

    # ------------------------------------------------------------------ 索引
    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    @property
    def index(self) -> ScenarioIndex:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> ScenarioIndex:
        if not self.root.is_dir():
            raise ScenarioLibraryError(f"病例库目录不存在：{self.root}")
        if not self.index_path.is_file():
            raise ScenarioLibraryError(f"病例库缺少索引文件：{self.index_path}")
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScenarioLibraryError(f"索引文件不是合法 JSON：{exc}") from exc
        try:
            index = ScenarioIndex.model_validate(raw)
        except ValidationError as exc:
            raise ScenarioLibraryError("索引文件不符合 Schema", detail=exc.errors()) from exc
        ids = [entry.scenario_id for entry in index.scenarios]
        if len(ids) != len(set(ids)):
            raise ScenarioLibraryError("索引里的 scenario_id 必须唯一")
        return index

    # ------------------------------------------------------------------ 读取
    def entries(self) -> list[ScenarioEntry]:
        return list(self.index.scenarios)

    def ids(self) -> list[str]:
        return [entry.scenario_id for entry in self.index.scenarios]

    def entry(self, scenario_id: str) -> ScenarioEntry:
        for item in self.index.scenarios:
            if item.scenario_id == scenario_id:
                return item
        raise ScenarioLibraryError(
            f"病例库里没有 {scenario_id}，可用：{', '.join(self.ids())}"
        )

    def path_for(self, scenario_id: str, version: str | None = None) -> Path:
        entry = self.entry(scenario_id)
        resolved = entry.resolve_version(version)
        return self.root / scenario_id / f"{resolved}.json"

    def load(self, scenario_id: str, version: str | None = None) -> PatientScenario:
        path = self.path_for(scenario_id, version)
        if not path.is_file():
            raise ScenarioLibraryError(f"病例文件不存在：{path}")
        cached = self._cache.get((scenario_id, path.stem))
        if cached is not None:
            return cached
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScenarioValidationError(f"{path.name} 不是合法 JSON：{exc}") from exc
        try:
            scenario = PatientScenario.model_validate(raw)
        except ValidationError as exc:
            raise ScenarioValidationError(
                f"{path.name} 不符合 PatientScenario Schema",
                detail=[{**item, "loc": list(item["loc"])} for item in exc.errors()],
            ) from exc
        if scenario.scenario_id != scenario_id:
            raise ScenarioValidationError(
                f"{path.name} 里的 scenario_id={scenario.scenario_id} 与目录 {scenario_id} 不一致"
            )
        declared = self.entry(scenario_id).versions
        if scenario.version not in declared:
            raise ScenarioLibraryError(
                f"{path.name} 的 version={scenario.version} 未在索引中登记"
            )
        self._cache[(scenario_id, path.stem)] = scenario
        return scenario

    def load_all(self, version: str | None = None) -> list[PatientScenario]:
        return [self.load(item.scenario_id, version) for item in self.entries()]
