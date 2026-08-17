# L4 治疗对话层（Treatment Dialogue Layer）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `D:\心理诊疗LLM\Layer 4` 下实现完整可运行、可测试的第四层治疗对话层 Python 包，将 L3 诊断层下发的 L1-L4 治疗指令"翻译"为自然语言治疗回复，并输出 L5 审核所需的 ContextualSafetyContext 与结构化临床摘要。

**Architecture:** 端口-适配器分层。核心领域（状态机、生成、回环A 编排）通过 LLMBackend/Reviewer 协议与具体实现（OpenAI 兼容 API / Mock / 本地模型占位）解耦；层间数据契约用 Pydantic 模型定义，字段与《五层认知架构》文档的 JSON 示例一一对应。L4 不做任何临床推理。

**Tech Stack:** Python 3.11+（实测 3.14.6）、Pydantic v2、pydantic-settings、openai（OpenAI 兼容接口）、FastAPI、uvicorn、pytest、httpx（TestClient）。

## Global Constraints

- 根目录：`D:\心理诊疗LLM\Layer 4`（所有路径相对此目录）
- Python >= 3.11；依赖：`pydantic>=2.12`、`pydantic-settings>=2.2`、`openai>=1.30`、`fastapi>=0.111`、`uvicorn[standard]>=0.30`；dev 依赖：`pytest>=8`、`httpx>=0.27`
- 包名 `l4`，目录 `l4/`，测试目录 `tests/`
- 测试一律使用 `MockBackend` + `MockReviewer`，禁止真实网络调用
- 每个任务结束必须跑测试并提交 git commit（Task 1 先 `git init`）
- 不做临床推理：回复生成仅翻译 L3 指令；即使 L3 指令有误也忠实执行
- `l4_substage` 枚举严格为 `"L4-L1" | "L4-L2" | "L4-L3" | "L4-L4"`
- 回环A：生成→审核→注入 fix_instruction 重写，最多 `max_rewrite_attempts`（默认 3）轮，超限输出兜底回复并标记 `loop_A_fallback` + `manual_review_required`
- 所有 Schema 的 `extra` 保持 Pydantic 默认（忽略未知字段），确保真实 L3/L5 的完整 JSON 可解析

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `l4/__init__.py`
- Create: `l4/config.py`
- Create: `tests/test_config.py`
- Create: `README.md`（先写占位，Task 9 补全）

**Interfaces:**
- Consumes: 无
- Produces: `l4.config.Settings`（字段：`llm_provider: Literal["openai_compat","mock","local"]="mock"`、`llm_base_url: str|None=None`、`llm_api_key: str|None=None`、`llm_model: str="deepseek-chat"`、`llm_temperature: float=0.7`、`review_mode: Literal["mock","l5_endpoint"]="mock"`、`max_rewrite_attempts: int=3`、`fallback_reply: str`）。环境变量前缀 `L4_`。Task 3 的 `get_backend(config)` 与 Task 7 的 `L4Service` 消费此对象。

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`：

```python
import os

import pytest

from l4.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.review_mode == "mock"
    assert s.llm_model == "deepseek-chat"
    assert s.max_rewrite_attempts == 3
    assert s.fallback_reply == "我可能需要重新整理一下思路，我们能换个角度聊吗？"


def test_env_override(monkeypatch):
    monkeypatch.setenv("L4_LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("L4_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("L4_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("L4_MAX_REWRITE_ATTEMPTS", "5")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai_compat"
    assert s.llm_base_url == "https://api.deepseek.com"
    assert s.llm_api_key == "sk-test"
    assert s.max_rewrite_attempts == 5


def test_invalid_provider():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="nope")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL（ModuleNotFoundError: l4）

- [ ] **Step 3: 最小实现**

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "l4-treatment-dialogue"
version = "0.1.0"
description = "五层认知架构 - 第四层治疗对话层（Treatment Dialogue Layer）"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.12",
    "pydantic-settings>=2.2",
    "openai>=1.30",
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "httpx>=0.27",
]

[tool.setuptools.packages.find]
include = ["l4*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`：

```gitignore
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.env
dist/
build/
.venv/
```

`.env.example`：

```dotenv
# LLM 提供商: mock | openai_compat | local
L4_LLM_PROVIDER=mock
# OpenAI 兼容接口（DeepSeek 示例）
L4_LLM_BASE_URL=https://api.deepseek.com
L4_LLM_API_KEY=
L4_LLM_MODEL=deepseek-chat
L4_LLM_TEMPERATURE=0.7
# 审核器: mock | l5_endpoint（l5_endpoint 尚未实现）
L4_REVIEW_MODE=mock
L4_MAX_REWRITE_ATTEMPTS=3
```

`l4/__init__.py`：

```python
__version__ = "0.1.0"
```

`l4/config.py`：

```python
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """L4 治疗对话层配置。环境变量前缀 L4_，例：L4_LLM_PROVIDER=mock。"""

    model_config = SettingsConfigDict(env_prefix="L4_", extra="ignore")

    llm_provider: Literal["openai_compat", "mock", "local"] = "mock"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7

    review_mode: Literal["mock", "l5_endpoint"] = "mock"
    max_rewrite_attempts: int = 3
    fallback_reply: str = "我可能需要重新整理一下思路，我们能换个角度聊吗？"
```

`README.md`（占位）：

```markdown
# Layer 4 治疗对话层

五层认知架构的第四层：接收 L3 诊断层下发的 L1-L4 治疗指令，将其"翻译"为自然语言治疗回复。本层不做临床推理。

完整说明见 Task 9 完成后补全。
```

- [ ] **Step 4: 安装并运行测试**

Run: `python -m pip install -e ".[dev]"`
Run: `python -m pytest tests/test_config.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git init
git add pyproject.toml .gitignore .env.example README.md l4 tests
git commit -m "feat: L4 项目脚手架与配置（Settings/pyproject/tests）"
```

---

### Task 2: 层间数据契约（schemas）

**Files:**
- Create: `l4/schemas/__init__.py`
- Create: `l4/schemas/instructions.py`
- Create: `l4/schemas/l3_report.py`
- Create: `l4/schemas/l5_context.py`
- Create: `l4/schemas/summary.py`
- Create: `l4/schemas/review.py`
- Create: `examples/sample_diagnostic_report.json`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: 无
- Produces（后续任务全部依赖这些类型）：
  - `Substage = Literal["L4-L1","L4-L2","L4-L3","L4-L4"]`（instructions.py）
  - `RoleInstruction(identity: str, style: str)`（instructions.py）
  - `DiagnosticAnchor(primary: str, comorbidities: list[str]=[], excluded: list[str]=[], core_beliefs: list[str]=[], plugins: list[str]=[], confidence: float=0.0, therapy_type: str="")`（instructions.py）
  - `TurnInstruction(goal: str, technique: str, forbidden: list[str]=[], link_previous: str="", plugin_guidance: str|None=None, force_substage: Substage|None=None)`（instructions.py）
  - `BoundaryInstruction(absolute_bans: list[str]=[], safety_trigger: str="")`（instructions.py）
  - `SafetyPassport(risk_level: int=1, sensitive_topics: list[str]=[], flags: list[str]=[])`（instructions.py）
  - `DiagnosticReport(...)`（l3_report.py），含 `to_anchor() -> DiagnosticAnchor`
  - `DiagnosisContext(primary: str="", secondary: list[str]=[], suicidal_ideation: str|None=None)`、`RiskContext(risk_level: int=1, sensitive_topics: list[str]=[])`、`L5Context(l4_raw_output: str, l4_substage: Substage, therapy_type: str, diagnosis_context: DiagnosisContext, risk_context: RiskContext)`（l5_context.py）
  - `EvidenceRef(source: str, finding: str, basis: str)`、`StructuredSummary(layer_id: Literal["L4"]="L4", timestamp: str, summary: str, structured_labels: dict, evidence_refs: list[EvidenceRef]=[], confidence: float=0.0, flags: list[str]=[])`（summary.py）
  - `Verdict = Literal["pass","fail"]`、`ReviewResult(harm: Verdict, boundary: Verdict, quality: Verdict, fail_reason: str="", fix_instruction: str="")` 含 `passed -> bool` 属性（review.py）

- [ ] **Step 1: 写失败测试**

`tests/test_schemas.py`：

```python
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from l4.schemas.review import ReviewResult
from l4.schemas.summary import EvidenceRef, StructuredSummary

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_role_instruction():
    r = RoleInstruction(identity="CBT治疗师", style="温暖")
    assert r.identity == "CBT治疗师"
    with pytest.raises(ValidationError):
        RoleInstruction()  # 缺少必填字段


def test_turn_instruction_force_substage_enum():
    t = TurnInstruction(goal="g", technique="t", force_substage="L4-L2")
    assert t.force_substage == "L4-L2"
    with pytest.raises(ValidationError):
        TurnInstruction(goal="g", technique="t", force_substage="L4-L5")


def test_anchor_therapy_type():
    a = DiagnosticAnchor(primary="MDD", therapy_type="CBT")
    assert a.therapy_type == "CBT"


def test_safety_passport_defaults():
    s = SafetyPassport()
    assert s.risk_level == 1
    assert s.sensitive_topics == []


def test_review_result_passed():
    ok = ReviewResult(harm="pass", boundary="pass", quality="pass")
    assert ok.passed is True
    bad = ReviewResult(harm="fail", boundary="pass", quality="pass",
                       fail_reason="x", fix_instruction="y")
    assert bad.passed is False


def test_l5_context_substage_enum():
    ctx = L5Context(l4_raw_output="你好", l4_substage="L4-L3",
                    therapy_type="CBT",
                    diagnosis_context=DiagnosisContext(primary="MDD"),
                    risk_context=RiskContext())
    assert ctx.l4_substage == "L4-L3"
    with pytest.raises(ValidationError):
        L5Context(l4_raw_output="你好", l4_substage="L4-L9",
                  therapy_type="CBT",
                  diagnosis_context=DiagnosisContext(),
                  risk_context=RiskContext())


def test_structured_summary_defaults():
    s = StructuredSummary(timestamp="2026-08-12T00:00:00Z",
                          summary="s",
                          structured_labels={},
                          evidence_refs=[EvidenceRef(source="a", finding="b", basis="c")],
                          confidence=0.85)
    assert s.layer_id == "L4"
    assert s.flags == []


def test_diagnostic_report_parses_sample_and_ignores_extra():
    raw = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    report = DiagnosticReport.model_validate(raw)
    assert report.diagnosis.primary.disorder == "MDD"
    assert report.diagnosis.primary.confidence == 0.87
    assert report.recommended_therapy.primary == "CBT"
    assert report.recommended_therapy.addon == ["CBT-I"]
    assert report.verifier_routing == "pass"
    anchor = report.to_anchor()
    assert anchor.therapy_type == "CBT"
    assert "CBT-I" in anchor.plugins
    assert "Insomnia Disorder" in anchor.comorbidities
    assert "Bipolar II" in anchor.excluded
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL（ModuleNotFoundError: l4.schemas）

- [ ] **Step 3: 最小实现**

`l4/schemas/__init__.py`：

```python
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from l4.schemas.review import ReviewResult
from l4.schemas.summary import EvidenceRef, StructuredSummary

__all__ = [
    "BoundaryInstruction",
    "DiagnosticAnchor",
    "DiagnosticReport",
    "DiagnosisContext",
    "EvidenceRef",
    "L5Context",
    "ReviewResult",
    "RiskContext",
    "RoleInstruction",
    "SafetyPassport",
    "StructuredSummary",
    "Substage",
    "TurnInstruction",
]
```

`l4/schemas/instructions.py`：

```python
from typing import Literal

from pydantic import BaseModel

Substage = Literal["L4-L1", "L4-L2", "L4-L3", "L4-L4"]


class RoleInstruction(BaseModel):
    """L1 角色定义（全会话复用）。"""

    identity: str
    style: str


class DiagnosticAnchor(BaseModel):
    """L2 诊断锚点（全会话复用，由 DiagnosticReport 派生）。"""

    primary: str = ""
    comorbidities: list[str] = []
    excluded: list[str] = []
    core_beliefs: list[str] = []
    plugins: list[str] = []
    confidence: float = 0.0
    therapy_type: str = ""


class TurnInstruction(BaseModel):
    """L3 本轮动态指令（每轮重新生成）。"""

    goal: str
    technique: str
    forbidden: list[str] = []
    link_previous: str = ""
    plugin_guidance: str | None = None
    force_substage: Substage | None = None


class BoundaryInstruction(BaseModel):
    """L4 防越界指令（每轮复用）。"""

    absolute_bans: list[str] = []
    safety_trigger: str = ""


class SafetyPassport(BaseModel):
    """L1 安全通行证中 L4 需要的部分（每轮复用）。"""

    risk_level: int = 1
    sensitive_topics: list[str] = []
    flags: list[str] = []
```

`l4/schemas/l3_report.py`：

```python
from pydantic import BaseModel

from l4.schemas.instructions import DiagnosticAnchor


class PrimaryDiagnosis(BaseModel):
    disorder: str
    confidence: float = 0.0
    severity: str = ""
    dsm5_criteria_met: list[str] = []


class SecondaryDiagnosis(BaseModel):
    disorder: str
    confidence: float = 0.0
    evidence: str = ""


class RuledOutDiagnosis(BaseModel):
    disorder: str
    reason: str = ""
    evidence_ref: str = ""


class Diagnosis(BaseModel):
    primary: PrimaryDiagnosis | None = None
    secondary: list[SecondaryDiagnosis] = []
    ruled_out: list[RuledOutDiagnosis] = []


class RecommendedTherapy(BaseModel):
    primary: str = ""
    addon: list[str] = []
    contraindications: list[str] = []


class TherapyRouting(BaseModel):
    primary: str = ""
    addon: list[str] = []
    l4_substage_priority: list[str] = []


class StructuredLabels(BaseModel):
    therapy_routing: TherapyRouting = TherapyRouting()
    wording_constraint: str = ""
    risk_monitoring: list[str] = []


class EvidenceRefIn(BaseModel):
    type: str = ""
    disorder: str = ""
    criteria_met: str = ""
    source_doc: str = ""
    threshold: str = ""
    therapy: str = ""
    source: str = ""


class DiagnosticReport(BaseModel):
    """接口三：L3 → L4 DiagnosticReport（诊断报告）。未知字段忽略，真实 L3 完整 JSON 可解析。"""

    layer_id: str = "L3"
    timestamp: str = ""
    diagnosis: Diagnosis = Diagnosis()
    verifier_score: float = 0.0
    verifier_routing: str = ""
    recommended_therapy: RecommendedTherapy = RecommendedTherapy()
    severity_level: int = 1
    clinical_notes_for_L4: str = ""
    summary: str = ""
    structured_labels: StructuredLabels = StructuredLabels()
    evidence_refs: list[EvidenceRefIn] = []
    confidence: float = 0.0
    flags: list[str] = []
    instruction_to_next: str = ""

    def to_anchor(self) -> DiagnosticAnchor:
        primary = self.diagnosis.primary
        return DiagnosticAnchor(
            primary=f"{primary.disorder}（置信度 {primary.confidence}）" if primary else "",
            comorbidities=[s.disorder for s in self.diagnosis.secondary],
            excluded=[r.disorder for r in self.diagnosis.ruled_out],
            core_beliefs=[],
            plugins=list(self.recommended_therapy.addon),
            confidence=primary.confidence if primary else 0.0,
            therapy_type=self.recommended_therapy.primary,
        )
```

`l4/schemas/l5_context.py`：

```python
from pydantic import BaseModel

from l4.schemas.instructions import Substage


class DiagnosisContext(BaseModel):
    primary: str = ""
    secondary: list[str] = []
    suicidal_ideation: str | None = None


class RiskContext(BaseModel):
    risk_level: int = 1
    sensitive_topics: list[str] = []


class L5Context(BaseModel):
    """接口四：L4 + L3 → L5 ContextualSafetyContext（上下文安全上下文）。"""

    l4_raw_output: str
    l4_substage: Substage
    therapy_type: str
    diagnosis_context: DiagnosisContext = DiagnosisContext()
    risk_context: RiskContext = RiskContext()
```

`l4/schemas/summary.py`：

```python
from typing import Any, Literal

from pydantic import BaseModel


class EvidenceRef(BaseModel):
    """循证依据引用：来源、发现、依据。"""

    source: str
    finding: str
    basis: str


class StructuredSummary(BaseModel):
    """每层输出附带的结构化临床摘要（通用格式，L4 填充）。"""

    layer_id: Literal["L4"] = "L4"
    timestamp: str
    summary: str
    structured_labels: dict[str, Any]
    evidence_refs: list[EvidenceRef] = []
    confidence: float = 0.0
    flags: list[str] = []
```

`l4/schemas/review.py`：

```python
from typing import Literal

from pydantic import BaseModel

Verdict = Literal["pass", "fail"]


class ReviewResult(BaseModel):
    """L5 审核结果（Mock 审核器与真实 L5 共用）。"""

    harm: Verdict
    boundary: Verdict
    quality: Verdict
    fail_reason: str = ""
    fix_instruction: str = ""

    @property
    def passed(self) -> bool:
        return self.harm == "pass" and self.boundary == "pass" and self.quality == "pass"
```

`examples/sample_diagnostic_report.json`（示例 L3 输入，基于架构文档示例字段）：

```json
{
  "layer_id": "L3",
  "timestamp": "2026-07-28T14:52:00Z",
  "diagnosis": {
    "primary": {
      "disorder": "MDD",
      "confidence": 0.87,
      "severity": "moderate",
      "dsm5_criteria_met": ["A1", "A2", "A3", "A4", "A5"]
    },
    "secondary": [
      {"disorder": "Insomnia Disorder", "confidence": 0.65, "evidence": "ISI=14接近重度阈值，入睡困难+早醒"}
    ],
    "ruled_out": [
      {"disorder": "Bipolar II", "reason": "MDQ=3<7阈值，无轻躁狂/躁狂发作史", "evidence_ref": "MDQ筛查指南"},
      {"disorder": "PTSD", "reason": "PCL-5 skipped per L1 safety flag", "evidence_ref": "L1 SafetyPassport"}
    ]
  },
  "verifier_score": 0.87,
  "verifier_routing": "pass",
  "recommended_therapy": {
    "primary": "CBT",
    "addon": ["CBT-I"],
    "contraindications": ["避免深度创伤暴露（per L1敏感话题标记）", "不推荐EMDR（无PTSD证据）"]
  },
  "severity_level": 1,
  "clinical_notes_for_L4": "患者核心认知模式推断为负性自动化思维（'我什么都做不好'）+回避行为（不愿出门、不愿社交）。L4治疗焦点：(1)行为激活——打破回避循环，重建日常活动节奏；(2)认知重构——识别和挑战负性自动化思维；(3)睡眠卫生+CBT-I技术（刺激控制+睡眠限制）。需温和监测自杀意念变化，若从被动转为主动，立即升级至L1安全协议。",
  "summary": "初步印象为中度MDD（置信度0.87，Verifier=0.87），共病失眠障碍（置信度0.65）。排除双相II型（MDQ=3）和PTSD（PCL-5跳过+临床怀疑度低）。推荐CBT联合CBT-I。",
  "structured_labels": {
    "therapy_routing": {
      "primary": "CBT",
      "addon": "CBT-I",
      "l4_substage_priority": ["behavioral_activation", "cognitive_restructuring", "sleep_hygiene"]
    },
    "wording_constraint": "使用'初步印象'措辞，禁止'确诊'",
    "risk_monitoring": ["passive_si_to_active_si_escalation"]
  },
  "evidence_refs": [
    {"type": "DSM-5", "disorder": "MDD", "criteria_met": "A1-A5 (5/9 >= 5)", "source_doc": "DSM-5_TR_zh_v1.0"},
    {"type": "scale", "disorder": "MDD", "source": "PHQ-9", "threshold": ">=15中重度"},
    {"type": "guideline", "therapy": "CBT_for_MDD", "source": "APA_2019; NICE_NG222"}
  ],
  "confidence": 0.87,
  "flags": ["passive_si_monitor", "trauma_sensitive", "insomnia_comorbid"],
  "instruction_to_next": "CBT对话应聚焦行为激活和认知重构，避免涉及创伤暴露练习。L4-L2需温和反馈诊断，征求患者确认。L4-L3按行为激活→认知重构→睡眠卫生顺序推进。"
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/schemas examples tests/test_schemas.py
git commit -m "feat: 层间数据契约 schemas（L1-L4指令/DiagnosticReport/L5Context/摘要/审核）"
```

---

### Task 3: LLM 后端抽象（端口 + 适配器 + 工厂）

**Files:**
- Create: `l4/llm/__init__.py`
- Create: `l4/llm/base.py`
- Create: `l4/llm/mock.py`
- Create: `l4/llm/openai_compat.py`
- Create: `l4/llm/local.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `l4.config.Settings`（Task 1）、`l4.schemas` 无需
- Produces:
  - `Message(role: Literal["system","user","assistant"], content: str)`（base.py）
  - `LLMBackend` Protocol：`chat(messages: list[Message], *, temperature: float = 0.7, json_mode: bool = False) -> str`（base.py）
  - `MockBackend(responses: dict[str,str] | None = None, default: str | None = None)`：`json_mode=True` 时按最后一条用户消息的子串匹配 `responses`，命中返回对应值，否则返回 `default`；`json_mode=False` 时直接返回匹配值或 `default`（mock.py）
  - `OpenAICompatBackend(api_key: str, base_url: str | None, model: str)`（openai_compat.py）
  - `LocalModelBackend`：构造即抛 `NotImplementedError`（local.py，占位）
  - `get_backend(config: Settings) -> LLMBackend`：按 `config.llm_provider` 分发（`__init__.py`）

- [ ] **Step 1: 写失败测试**

`tests/test_llm.py`：

```python
import pytest

from l4.config import Settings
from l4.llm import get_backend
from l4.llm.base import Message
from l4.llm.local import LocalModelBackend
from l4.llm.mock import MockBackend
from l4.llm.openai_compat import OpenAICompatBackend


def test_mock_json_mode_matches_substring():
    backend = MockBackend(
        responses={"触发词": '{"reply": "违规回复", "stage_complete": false, "reason": "x"}'},
        default='{"reply": "正常回复", "stage_complete": true, "reason": "y"}',
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="这里包含触发词，请回复"),
    ]
    assert backend.chat(messages, json_mode=True) == '{"reply": "违规回复", "stage_complete": false, "reason": "x"}'

    other = [Message(role="user", content="没有触发词")]
    assert backend.chat(other, json_mode=True) == '{"reply": "正常回复", "stage_complete": true, "reason": "y"}'


def test_mock_non_json_returns_raw():
    backend = MockBackend(default="普通文本回复")
    out = backend.chat([Message(role="user", content="hi")], json_mode=False)
    assert out == "普通文本回复"


def test_factory_dispatch():
    assert isinstance(get_backend(Settings(_env_file=None, llm_provider="mock")), MockBackend)
    assert isinstance(
        get_backend(Settings(_env_file=None, llm_provider="openai_compat",
                             llm_api_key="sk-x", llm_base_url="https://api.deepseek.com",
                             llm_model="deepseek-chat")),
        OpenAICompatBackend,
    )


def test_factory_local_raises():
    with pytest.raises(NotImplementedError):
        get_backend(Settings(_env_file=None, llm_provider="local"))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL（ModuleNotFoundError: l4.llm）

- [ ] **Step 3: 最小实现**

`l4/llm/__init__.py`：

```python
from l4.config import Settings
from l4.llm.base import LLMBackend
from l4.llm.local import LocalModelBackend
from l4.llm.mock import MockBackend
from l4.llm.openai_compat import OpenAICompatBackend


def get_backend(config: Settings) -> LLMBackend:
    """按配置返回 LLM 后端实例。后期本地模型就绪时，实现 l4/llm/local.py 即可。"""
    if config.llm_provider == "mock":
        return MockBackend()
    if config.llm_provider == "openai_compat":
        return OpenAICompatBackend(
            api_key=config.llm_api_key or "",
            base_url=config.llm_base_url,
            model=config.llm_model,
        )
    if config.llm_provider == "local":
        return LocalModelBackend()
    raise ValueError(f"未知 LLM 提供商: {config.llm_provider}")
```

`l4/llm/base.py`：

```python
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


@runtime_checkable
class LLMBackend(Protocol):
    """LLM 后端端口。所有适配器（API/Mock/本地模型）实现此协议。"""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """messages 为 [{role, content}] 消息列表；json_mode=True 时要求返回合法 JSON 文本。"""
        ...


class LLMError(Exception):
    pass
```

`l4/llm/mock.py`：

```python
from l4.llm.base import LLMBackend, Message

DEFAULT_MOCK_RESPONSE = (
    '{"reply": "我听到你说的了，这很不容易。我们可以一起看看现在的情况。"'
    ', "stage_complete": false, "reason": "mock 默认回复"}'
)


class MockBackend(LLMBackend):
    """FakeLLM：按最后一条用户消息子串匹配预设回复。无密钥可跑测试与演示。"""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default or DEFAULT_MOCK_RESPONSE

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        for key, value in self._responses.items():
            if key and key in last_user:
                return value
        return self._default
```

`l4/llm/openai_compat.py`：

```python
from openai import OpenAI

from l4.llm.base import LLMBackend, Message


class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容接口适配器（DeepSeek 等）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        payload: list[dict[str, str]] = [m.model_dump() for m in messages]
        kwargs: dict = {"model": self._model, "messages": payload, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content or ""
```

`l4/llm/local.py`：

```python
from l4.llm.base import LLMBackend, Message


class LocalModelBackend(LLMBackend):
    """本地训练模型适配器占位。

    TODO: 本地模型训练完成后，在此实现 vLLM / transformers 接入
    （如 OpenAI 兼容的 vLLM server 或 transformers pipeline），
    并保持 chat() 签名不变。L4 其他代码无需任何改动。
    """

    def __init__(self, model_path: str = "") -> None:
        raise NotImplementedError(
            "LocalModelBackend 尚未实现：本地模型训练完成后接入 vLLM/transformers。"
        )

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/llm tests/test_llm.py
git commit -m "feat: LLM 后端抽象（Mock/OpenAI兼容/本地占位 + 工厂）"
```

---

### Task 4: 会话状态机与历史存储

**Files:**
- Create: `l4/session/__init__.py`
- Create: `l4/session/state_machine.py`
- Create: `l4/session/history.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `Substage`、`TurnInstruction`（Task 2）、`Message`（Task 3）
- Produces:
  - `SUBSTAGE_ORDER: tuple[Substage, ...]`（state_machine.py）
  - `SubstageMachine(BaseModel)`：字段 `current: Substage = "L4-L1"`、`flags: list[str] = []`；方法 `resolve(force: Substage|None=None) -> Substage`、`advance(stage_complete: bool=False, force: Substage|None=None) -> None`（state_machine.py）
  - `ConversationTurn(role: Literal["user","assistant"], content: str)`、`SessionData(session_id, machine, history, l3_history, role, anchor, boundary, safety, therapy_options, meta)` 含 `get_messages() -> list[Message]`、`SessionStore` 含 `create/get/exists/append_turn/append_l3`（history.py）
- 推进规则：`resolve` 返回 `force or current`；`advance` 时若 `force` 非空直接设为 `current`（回退时打 `stage_regression_override` 标记）；否则 `stage_complete=True` 时前进一步；终态再次收到完成信号打 `terminal_reached` 标记；标记去重

- [ ] **Step 1: 写失败测试**

`tests/test_session.py`：

```python
import pytest

from l4.llm.base import Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.history import ConversationTurn, SessionData, SessionStore
from l4.session.state_machine import SUBSTAGE_ORDER, SubstageMachine


def test_substage_order():
    assert SUBSTAGE_ORDER == ("L4-L1", "L4-L2", "L4-L3", "L4-L4")


def test_default_forward_progression():
    m = SubstageMachine()
    assert m.resolve() == "L4-L1"
    m.advance(stage_complete=True)
    assert m.current == "L4-L2"
    m.advance(stage_complete=True)
    assert m.current == "L4-L3"
    m.advance(stage_complete=True)
    assert m.current == "L4-L4"
    m.advance(stage_complete=True)
    assert m.current == "L4-L4"  # 终态不越界
    assert "terminal_reached" in m.flags


def test_force_substage_overrides():
    m = SubstageMachine(current="L4-L1")
    assert m.resolve(force="L4-L3") == "L4-L3"
    m.advance(force="L4-L3")
    assert m.current == "L4-L3"


def test_regression_override_flagged():
    m = SubstageMachine(current="L4-L3")
    m.advance(force="L4-L1")
    assert m.current == "L4-L1"
    assert "stage_regression_override" in m.flags


def test_not_complete_no_advance():
    m = SubstageMachine(current="L4-L2")
    m.advance(stage_complete=False)
    assert m.current == "L4-L2"


def test_flags_deduplicated():
    m = SubstageMachine(current="L4-L1")
    m.advance(force="L4-L1")
    m.advance(force="L4-L1")
    assert m.flags.count("stage_regression_override") == 1


def _make_session_data(session_id="s1") -> SessionData:
    return SessionData(
        session_id=session_id,
        machine=SubstageMachine(),
        role=RoleInstruction(identity="CBT治疗师", style="温暖"),
        anchor=None,
        boundary=BoundaryInstruction(),
        safety=SafetyPassport(),
        therapy_options=["CBT", "CBT-I"],
    )


def test_session_store_lifecycle():
    store = SessionStore()
    assert not store.exists("s1")
    data = _make_session_data()
    store.create(data)
    assert store.exists("s1")
    assert store.get("s1").session_id == "s1"
    with pytest.raises(KeyError):
        store.get("nope")


def test_session_get_messages_and_history():
    data = _make_session_data()
    data.history.append(ConversationTurn(role="user", content="你好"))
    data.history.append(ConversationTurn(role="assistant", content="你好，今天想聊点什么？"))
    msgs = data.get_messages()
    assert msgs == [
        Message(role="user", content="你好"),
        Message(role="assistant", content="你好，今天想聊点什么？"),
    ]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_session.py -v`
Expected: FAIL（ModuleNotFoundError: l4.session）

- [ ] **Step 3: 最小实现**

`l4/session/__init__.py`：

```python
from l4.session.history import SessionData, SessionStore
from l4.session.state_machine import SUBSTAGE_ORDER, SubstageMachine

__all__ = ["SUBSTAGE_ORDER", "SessionData", "SessionStore", "SubstageMachine"]
```

`l4/session/state_machine.py`：

```python
from typing import Literal

from pydantic import BaseModel

from l4.schemas.instructions import Substage

SUBSTAGE_ORDER: tuple[Substage, ...] = ("L4-L1", "L4-L2", "L4-L3", "L4-L4")


class SubstageMachine(BaseModel):
    """L4 子层状态机：L4-L1 共情 → L4-L2 结果反馈 → L4-L3 技术执行 → L4-L4 作业布置。

    推进规则（三路信号取最高优先级）：
    1. force（L3 强制覆盖）非空 → 直接跳转，回退时打 stage_regression_override；
    2. stage_complete=True（阶段完成判定）→ 前进一步；
    3. 其余情况保持当前阶段。
    """

    current: Substage = "L4-L1"
    flags: list[str] = []

    def resolve(self, force: Substage | None = None) -> Substage:
        """返回本轮应使用的子阶段（不推进）。"""
        return force or self.current

    def advance(self, stage_complete: bool = False, force: Substage | None = None) -> None:
        if force is not None:
            if SUBSTAGE_ORDER.index(force) < SUBSTAGE_ORDER.index(self.current):
                self._flag("stage_regression_override")
            self.current = force
            return
        idx = SUBSTAGE_ORDER.index(self.current)
        if stage_complete and idx < len(SUBSTAGE_ORDER) - 1:
            self.current = SUBSTAGE_ORDER[idx + 1]
        elif stage_complete and idx == len(SUBSTAGE_ORDER) - 1:
            self._flag("terminal_reached")

    def _flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
```

`l4/session/history.py`：

```python
from typing import Literal

from pydantic import BaseModel

from l4.llm.base import Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.state_machine import SubstageMachine


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionData(BaseModel):
    session_id: str
    machine: SubstageMachine = SubstageMachine()
    history: list[ConversationTurn] = []
    l3_history: list[TurnInstruction] = []
    role: RoleInstruction = RoleInstruction(identity="", style="")
    anchor: DiagnosticAnchor | None = None
    boundary: BoundaryInstruction = BoundaryInstruction()
    safety: SafetyPassport = SafetyPassport()
    therapy_options: list[str] = []
    meta: dict = {}

    def get_messages(self) -> list[Message]:
        return [Message(role=t.role, content=t.content) for t in self.history]


class SessionStore:
    """内存会话存储：按 session_id 维护 SessionData。"""

    def __init__(self) -> None:
        self._data: dict[str, SessionData] = {}

    def create(self, data: SessionData) -> None:
        self._data[data.session_id] = data

    def get(self, session_id: str) -> SessionData:
        return self._data[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._data
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_session.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/session tests/test_session.py
git commit -m "feat: L4 子层状态机与会话历史存储"
```

---

### Task 5: Prompt 组装与生成

**Files:**
- Create: `l4/generation/__init__.py`
- Create: `l4/generation/prompt_assembler.py`
- Create: `l4/generation/responder.py`
- Create: `l4/generation/summary_builder.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `RoleInstruction/DiagnosticAnchor/BoundaryInstruction/TurnInstruction/SafetyPassport`（Task 2）、`L5Context/DiagnosisContext/RiskContext`（Task 2）、`StructuredSummary/EvidenceRef`（Task 2）、`LLMBackend/Message`（Task 3）
- Produces:
  - `assemble_prompt(role, anchor, boundary, turn, history: list[Message], user_message, fix_instruction: str|None=None) -> list[Message]`（prompt_assembler.py）：固定返回 `[system 消息, user 消息]`；system 含 L1/L2/L4 三段；user 含 本轮(L3+插件)、上下文、当前消息；`fix_instruction` 追加在 user 内容末尾
  - `GenerationResult(reply: str, stage_complete: bool=False, reason: str="")`、`Responder(backend: LLMBackend, role, anchor, boundary)` 方法 `generate(turn, history, user_message, fix_instruction=None) -> GenerationResult`：优先 `json_mode=True` 解析 JSON，失败重试一次 `json_mode=False` 取纯文本（responder.py）
  - `build_l5_context(reply, substage, therapy_type, safety, anchor) -> L5Context`、`build_evidence_refs(turn, history, safety) -> list[EvidenceRef]`、`build_summary(turn, substage, reply, evidence_refs, confidence=0.85, flags=[]) -> StructuredSummary`（summary_builder.py）

- [ ] **Step 1: 写失败测试**

`tests/test_generation.py`：

```python
from l4.generation.prompt_assembler import assemble_prompt
from l4.generation.responder import GenerationResult, Responder
from l4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)
from l4.llm.base import Message
from l4.llm.mock import MockBackend
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.history import ConversationTurn

ROLE = RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化")
ANCHOR = DiagnosticAnchor(primary="MDD（置信度 0.87）", comorbidities=["Insomnia Disorder"],
                          excluded=["Bipolar II", "PTSD"], plugins=["CBT-I"],
                          confidence=0.87, therapy_type="CBT")
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来"],
                               safety_trigger="用户表达自杀意图→立即触发安全协议")


def _turn(**kw) -> TurnInstruction:
    defaults = dict(goal="探索自动化思维", technique="苏格拉底式提问",
                    forbidden=["建议", "确诊"], link_previous="上次你提到……")
    defaults.update(kw)
    return TurnInstruction(**defaults)


def test_assemble_prompt_four_sections():
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(), history=[], user_message="我这周又加班到凌晨")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    sys_text = msgs[0].content
    # L1 角色
    assert "你是一名受过CBT训练的心理治疗师" in sys_text
    # L2 诊断锚点
    assert "MDD（置信度 0.87）" in sys_text
    # L4 防越界
    assert "禁止开药" in sys_text
    user_text = msgs[1].content
    # L3 本轮指令
    assert "苏格拉底式提问" in user_text
    # 上下文与当前消息
    assert "我这周又加班到凌晨" in user_text


def test_assemble_prompt_history_and_plugin():
    history = [Message(role="user", content="你好"),
               Message(role="assistant", content="今天想聊什么？")]
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(plugin_guidance="本轮开场嵌入3分钟呼吸空间"),
                           history=history, user_message="我睡不着")
    user_text = msgs[1].content
    assert "3分钟呼吸空间" in user_text
    assert "今天想聊什么" in user_text


def test_assemble_prompt_fix_instruction_appended_at_end():
    msgs = assemble_prompt(role=ROLE, anchor=ANCHOR, boundary=BOUNDARY,
                           turn=_turn(), history=[], user_message="hi",
                           fix_instruction="删除确定性措辞，改为初步印象")
    user_text = msgs[1].content
    assert "删除确定性措辞，改为初步印象" in user_text
    assert user_text.strip().endswith("删除确定性措辞，改为初步印象")


def test_responder_parses_json():
    backend = MockBackend(
        default='{"reply": "听起来工作给你带来了很多负担。", "stage_complete": true, "reason": "目标达成"}'
    )
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="最近压力很大")
    assert gen == GenerationResult(reply="听起来工作给你带来了很多负担。",
                                   stage_complete=True, reason="目标达成")


def test_responder_json_parse_failure_falls_back_to_text():
    backend = MockBackend(default="这不是JSON，只是一句普通的回复")
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="hi")
    assert gen.reply == "这不是JSON，只是一句普通的回复"
    assert gen.stage_complete is False
    assert gen.reason == "parse_fallback"


def test_responder_forwards_fix_instruction():
    backend = MockBackend(responses={"删除确定性措辞": '{"reply": "改写后回复", "stage_complete": false, "reason": ""}'})
    resp = Responder(backend, ROLE, ANCHOR, BOUNDARY)
    gen = resp.generate(_turn(), history=[], user_message="hi",
                        fix_instruction="删除确定性措辞，改为初步印象")
    assert gen.reply == "改写后回复"


def test_build_l5_context_fields():
    safety = SafetyPassport(risk_level=2, sensitive_topics=["trauma"],
                            flags=["monitor_suicidal_ideation"])
    ctx = build_l5_context("你好", "L4-L2", "CBT", safety, ANCHOR)
    assert ctx.l4_raw_output == "你好"
    assert ctx.l4_substage == "L4-L2"
    assert ctx.therapy_type == "CBT"
    assert ctx.diagnosis_context.primary == "MDD（置信度 0.87）"
    assert ctx.diagnosis_context.secondary == ["Insomnia Disorder"]
    assert ctx.diagnosis_context.suicidal_ideation == "passive_only"
    assert ctx.risk_context.risk_level == 2
    assert ctx.risk_context.sensitive_topics == ["trauma"]


def test_build_evidence_refs_non_empty():
    history = [ConversationTurn(role="user", content="我躺床上两三个小时睡不着"),
               ConversationTurn(role="assistant", content="听起来很辛苦")]
    refs = build_evidence_refs(_turn(), history,
                               SafetyPassport(flags=["monitor_suicidal_ideation"]))
    sources = {r.source for r in refs}
    assert "L3_turn_instruction" in sources
    assert "conversation_history" in sources
    assert "L1_safety_passport" in sources


def test_build_summary_fields():
    refs = [{"source": "L3_turn_instruction", "finding": "technique=苏格拉底式提问", "basis": "目标"}]
    s = build_summary(_turn(), "L4-L3", "测试回复", refs, confidence=0.85,
                      flags=["stage_regression_override"])
    assert s.layer_id == "L4"
    assert s.structured_labels["substage"] == "L4-L3"
    assert s.structured_labels["goal"] == "探索自动化思维"
    assert s.confidence == 0.85
    assert "stage_regression_override" in s.flags
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_generation.py -v`
Expected: FAIL（ModuleNotFoundError: l4.generation）

- [ ] **Step 3: 最小实现**

`l4/generation/__init__.py`：

```python
from l4.generation.prompt_assembler import assemble_prompt
from l4.generation.responder import GenerationResult, Responder
from l4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)

__all__ = [
    "GenerationResult",
    "Responder",
    "assemble_prompt",
    "build_evidence_refs",
    "build_l5_context",
    "build_summary",
]
```

`l4/generation/prompt_assembler.py`：

```python
from l4.llm.base import Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    TurnInstruction,
)

_SYSTEM_INTRO = (
    "你是一名受过CBT训练的心理治疗师。你的唯一任务是把诊断层给出的治疗指令"
    "翻译为自然、温暖、专业的治疗对话。\n"
    "你是语言生成引擎，不做任何临床推理：不诊断、不调整治疗方案、不评判指令。"
)


def assemble_prompt(
    *,
    role: RoleInstruction,
    anchor: DiagnosticAnchor,
    boundary: BoundaryInstruction,
    turn: TurnInstruction,
    history: list[Message],
    user_message: str,
    fix_instruction: str | None = None,
) -> list[Message]:
    """四段式组装 prompt（文档第 6 节）：系统(L1+L2+L4) / 本轮(L3+插件) / 上下文 / 当前。"""
    system_parts = [_SYSTEM_INTRO]
    system_parts.append(f"【角色定义】(L1)\n{role.identity}\n风格：{role.style}")
    system_parts.append(
        "【诊断锚点】(L2)\n"
        f"主诊断：{anchor.primary}\n"
        f"共病：{', '.join(anchor.comorbidities) if anchor.comorbidities else '无'}\n"
        f"已排除：{', '.join(anchor.excluded) if anchor.excluded else '无'}\n"
        f"核心信念：{', '.join(anchor.core_beliefs) if anchor.core_beliefs else '无'}\n"
        f"已激活插件：{', '.join(anchor.plugins) if anchor.plugins else '无'}\n"
        f"诊断置信度：{anchor.confidence}"
    )
    bans = "\n".join(f"- {b}" for b in boundary.absolute_bans) or "- 无"
    system_parts.append(
        "【防越界指令】(L4)\n"
        f"绝对禁止：\n{bans}\n"
        f"安全触发：{boundary.safety_trigger or '无'}"
    )
    system_parts.append(
        '【输出格式】请只输出一个 JSON 对象，不要包含任何其他文字，格式：\n'
        '{"reply": "治疗师回复正文", "stage_complete": true或false, "reason": "本轮目标是否达成的依据"}'
    )

    user_parts = [
        "【本轮指令】(L3)",
        f"目标：{turn.goal}",
        f"技术：{turn.technique}",
        f"禁止：{', '.join(turn.forbidden) if turn.forbidden else '无'}",
        f"衔接上轮：{turn.link_previous or '无'}",
        "插件嵌入指导：",
        turn.plugin_guidance or "无",
    ]
    if history:
        lines = [f"{'用户' if m.role == 'user' else '治疗师'}：{m.content}" for m in history]
        user_parts.append("【对话历史】\n" + "\n".join(lines))
    else:
        user_parts.append("【对话历史】（无）")
    user_parts.append(f"【当前消息】\n用户：{user_message}")
    user_parts.append("请生成治疗师回复（JSON）。")
    if fix_instruction:
        user_parts.append(f"【重写要求（审核反馈）】\n{fix_instruction}\n请根据以上要求重写回复，输出 JSON。")

    return [
        Message(role="system", content="\n\n".join(system_parts)),
        Message(role="user", content="\n\n".join(user_parts)),
    ]
```

`l4/generation/responder.py`：

```python
import json

from pydantic import BaseModel, ValidationError

from l4.generation.prompt_assembler import assemble_prompt
from l4.llm.base import LLMBackend, Message
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    TurnInstruction,
)


class GenerationResult(BaseModel):
    reply: str
    stage_complete: bool = False
    reason: str = ""


class Responder:
    """调用 LLM 生成治疗回复并解析输出。"""

    def __init__(
        self,
        backend: LLMBackend,
        role: RoleInstruction,
        anchor: DiagnosticAnchor,
        boundary: BoundaryInstruction,
    ) -> None:
        self._backend = backend
        self._role = role
        self._anchor = anchor
        self._boundary = boundary

    def generate(
        self,
        turn: TurnInstruction,
        history: list[Message],
        user_message: str,
        fix_instruction: str | None = None,
    ) -> GenerationResult:
        messages = assemble_prompt(
            role=self._role, anchor=self._anchor, boundary=self._boundary,
            turn=turn, history=history, user_message=user_message,
            fix_instruction=fix_instruction,
        )
        raw = self._backend.chat(messages, json_mode=True)
        try:
            parsed = json.loads(raw)
            return GenerationResult.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, TypeError):
            fallback = self._backend.chat(messages, json_mode=False)
            return GenerationResult(reply=fallback, stage_complete=False,
                                    reason="parse_fallback")
```

`l4/generation/summary_builder.py`：

```python
from datetime import datetime, timezone

from l4.schemas.instructions import (
    DiagnosticAnchor,
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from l4.schemas.l5_context import DiagnosisContext, L5Context, RiskContext
from l4.schemas.summary import EvidenceRef, StructuredSummary
from l4.session.history import ConversationTurn


def build_l5_context(
    reply: str,
    substage: Substage,
    therapy_type: str,
    safety: SafetyPassport,
    anchor: DiagnosticAnchor,
) -> L5Context:
    suicidal = "passive_only" if "monitor_suicidal_ideation" in safety.flags else None
    return L5Context(
        l4_raw_output=reply,
        l4_substage=substage,
        therapy_type=therapy_type,
        diagnosis_context=DiagnosisContext(
            primary=anchor.primary,
            secondary=anchor.comorbidities,
            suicidal_ideation=suicidal,
        ),
        risk_context=RiskContext(
            risk_level=safety.risk_level,
            sensitive_topics=safety.sensitive_topics,
        ),
    )


def build_evidence_refs(
    turn: TurnInstruction,
    history: list[ConversationTurn],
    safety: SafetyPassport,
) -> list[EvidenceRef]:
    refs = [
        EvidenceRef(
            source="L3_turn_instruction",
            finding=f"technique={turn.technique}",
            basis=f"本轮目标：{turn.goal}",
        )
    ]
    if history:
        last_user = next(
            (t.content for t in reversed(history) if t.role == "user"), ""
        )
        refs.append(
            EvidenceRef(
                source="conversation_history",
                finding=f"{len(history)} 条历史消息",
                basis=f"最近用户消息：{last_user[:50]}",
            )
        )
    if safety.flags:
        refs.append(
            EvidenceRef(
                source="L1_safety_passport",
                finding=f"risk_level={safety.risk_level}",
                basis="；".join(safety.flags),
            )
        )
    return refs


def build_summary(
    turn: TurnInstruction,
    substage: Substage,
    reply: str,
    evidence_refs: list[EvidenceRef],
    confidence: float = 0.85,
    flags: list[str] | None = None,
) -> StructuredSummary:
    return StructuredSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary=reply[:200],
        structured_labels={
            "goal": turn.goal,
            "technique": turn.technique,
            "substage": substage,
            "plugin_guidance": turn.plugin_guidance or "",
        },
        evidence_refs=evidence_refs,
        confidence=confidence,
        flags=list(flags or []),
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_generation.py -v`
Expected: PASS（10 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/generation tests/test_generation.py
git commit -m "feat: prompt 四段式组装、回复生成与结构化摘要构建"
```

---

### Task 6: 审核端口与回环A 编排

**Files:**
- Create: `l4/review/__init__.py`
- Create: `l4/review/reviewer.py`
- Create: `l4/review/mock_reviewer.py`
- Create: `l4/review/orchestrator.py`
- Test: `tests/test_review_loop.py`

**Interfaces:**
- Consumes: `ReviewResult`（Task 2）、`L5Context`（Task 2）、`Responder/GenerationResult`（Task 5）、`Message/TurnInstruction`（Task 2/3）
- Produces:
  - `Reviewer` Protocol：`review(ctx: L5Context, review_history: list[ReviewResult]) -> ReviewResult`（reviewer.py）
  - `MockReviewer(fail_first: int = 0, always_fail: bool = False, fail_trigger: str = "")`（mock_reviewer.py）：`fail_trigger` 命中输出、前 `fail_first` 次、`always_fail` 均返回 fail（含 fail_reason + fix_instruction），否则 pass
  - `ReviewOutcome(reply: str, review_history: list[ReviewResult], attempts: int, fallback_used: bool, fix_instructions: list[str], stage_complete: bool=False)`、`ReviewOrchestrator(responder, reviewer, max_attempts=3, fallback_reply=DEFAULT_FALLBACK)` 方法 `run(turn, history, user_message, build_ctx: Callable[[str], L5Context], validate: Callable[[str], list[str]] | None = None, initial_fix: str | None = None) -> ReviewOutcome`（orchestrator.py）
  - `DEFAULT_FALLBACK = "我可能需要重新整理一下思路，我们能换个角度聊吗？"`（orchestrator.py）

- [ ] **Step 1: 写失败测试**

`tests/test_review_loop.py`：

```python
from l4.generation.responder import Responder
from l4.generation.summary_builder import build_l5_context
from l4.llm.base import Message
from l4.llm.mock import MockBackend
from l4.review.mock_reviewer import MockReviewer
from l4.review.orchestrator import DEFAULT_FALLBACK, ReviewOrchestrator
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.session.history import ConversationTurn

ROLE = RoleInstruction(identity="CBT治疗师", style="温暖")
ANCHOR = DiagnosticAnchor(primary="MDD", confidence=0.87, therapy_type="CBT")
BOUNDARY = BoundaryInstruction()
SAFETY = SafetyPassport()

TURN = TurnInstruction(goal="探索自动化思维", technique="苏格拉底式提问")

GOOD = '{"reply": "听起来这周很辛苦，是什么让你觉得自己必须这么做？", "stage_complete": true, "reason": "ok"}'
BAD = '{"reply": "你得了中度抑郁，建议你试试加大安眠药剂量。", "stage_complete": false, "reason": "bad"}'


def _make_orchestrator(mock_backend, mock_reviewer=None, max_attempts=3):
    responder = Responder(mock_backend, ROLE, ANCHOR, BOUNDARY)
    return ReviewOrchestrator(responder, mock_reviewer or MockReviewer(), max_attempts)


def _run(orchestrator, user_message="我这周又加班到凌晨，感觉撑不住了", validate=None):
    def build_ctx(reply):
        return build_l5_context(reply, "L4-L3", "CBT", SAFETY, ANCHOR)
    return orchestrator.run(TURN, [], user_message, build_ctx, validate)


def test_passes_first_try():
    orch = _make_orchestrator(MockBackend(default=GOOD))
    out = _run(orch)
    assert out.fallback_used is False
    assert out.attempts == 1
    assert len(out.review_history) == 1
    assert out.review_history[0].passed is True
    assert "觉得自己必须这么做" in out.reply


def test_fail_then_pass_uses_fix_instruction():
    backend = MockBackend(
        responses={
            "重写要求": '{"reply": "根据我们目前的评估，初步印象是情绪困扰，你感觉呢？", "stage_complete": false, "reason": "rewritten"}'
        },
        default=BAD,
    )
    orch = _make_orchestrator(backend, MockReviewer(fail_first=1))
    out = _run(orch)
    assert out.fallback_used is False
    assert out.attempts == 2
    assert len(out.fix_instructions) == 1
    assert "初步印象" in out.reply


def test_three_fails_fallback():
    backend = MockBackend(
        responses={"重写要求": BAD},
        default=BAD,
    )
    orch = _make_orchestrator(backend, MockReviewer(always_fail=True), max_attempts=3)
    out = _run(orch)
    assert out.fallback_used is True
    assert out.attempts == 3
    assert out.reply == DEFAULT_FALLBACK
    assert len(out.review_history) == 3
    assert all(not r.passed for r in out.review_history)


def test_validate_hook_fails_without_reviewer_fail():
    backend = MockBackend(
        responses={
            "请修正": '{"reply": "安全回复，不含禁止词。", "stage_complete": false, "reason": ""}'
        },
        default='{"reply": "这里出现了禁止词：建议", "stage_complete": false, "reason": ""}',
    )
    orch = _make_orchestrator(backend, MockReviewer(), max_attempts=2)

    def validate(reply: str) -> list[str]:
        return ["回复包含禁止词：建议"] if "建议" in reply else []

    out = _run(orch, validate=validate)
    assert out.fallback_used is False
    assert out.attempts == 2
    assert "安全回复" in out.reply
    assert "本地校验未通过" in out.review_history[0].fail_reason


def test_initial_fix_injected():
    backend = MockBackend(responses={"删除确定性措辞": GOOD}, default=BAD)
    orch = _make_orchestrator(backend, MockReviewer())
    out = _run(orch)
    assert out.fallback_used is False
    assert "删除确定性措辞" in backend._responses  # 由 initial_fix 触发匹配
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_review_loop.py -v`
Expected: FAIL（ModuleNotFoundError: l4.review）

- [ ] **Step 3: 最小实现**

`l4/review/__init__.py`：

```python
from l4.review.mock_reviewer import MockReviewer
from l4.review.orchestrator import DEFAULT_FALLBACK, ReviewOrchestrator, ReviewOutcome

__all__ = ["DEFAULT_FALLBACK", "MockReviewer", "ReviewOrchestrator", "ReviewOutcome"]
```

`l4/review/reviewer.py`：

```python
from typing import Protocol, runtime_checkable

from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult


@runtime_checkable
class Reviewer(Protocol):
    """审核端口：真实 L5 与 Mock 审核器共用此协议。"""

    def review(
        self,
        ctx: L5Context,
        review_history: list[ReviewResult],
    ) -> ReviewResult:
        """审核一条 L4 输出，返回 harm/boundary/quality 判定。"""
        ...
```

`l4/review/mock_reviewer.py`：

```python
from l4.review.reviewer import Reviewer
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult


class MockReviewer(Reviewer):
    """可脚本化审核器：默认全通过；fail_first 前 N 次失败；fail_trigger 命中失败；always_fail 恒失败。"""

    def __init__(
        self,
        fail_first: int = 0,
        always_fail: bool = False,
        fail_trigger: str = "",
    ) -> None:
        self._fail_first = fail_first
        self._always_fail = always_fail
        self._fail_trigger = fail_trigger

    def review(
        self,
        ctx: L5Context,
        review_history: list[ReviewResult],
    ) -> ReviewResult:
        failed = (
            self._always_fail
            or len(review_history) < self._fail_first
            or (self._fail_trigger and self._fail_trigger in ctx.l4_raw_output)
        )
        if not failed:
            return ReviewResult(harm="pass", boundary="pass", quality="pass")
        return ReviewResult(
            harm="fail",
            boundary="fail",
            quality="fail",
            fail_reason=(
                f"Mock审核未通过（触发：{self._fail_trigger or '脚本配置'}，"
                f"已尝试 {len(review_history) + 1} 次）"
            ),
            fix_instruction="请改写回复：删除违规内容，使用协作式、非确定性的措辞，并保留共情。",
        )
```

`l4/review/orchestrator.py`：

```python
from collections.abc import Callable

from pydantic import BaseModel

from l4.generation.responder import Responder
from l4.llm.base import Message
from l4.review.reviewer import Reviewer
from l4.schemas.instructions import TurnInstruction
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult

DEFAULT_FALLBACK = "我可能需要重新整理一下思路，我们能换个角度聊吗？"


class ReviewOutcome(BaseModel):
    reply: str
    review_history: list[ReviewResult]
    attempts: int
    fallback_used: bool
    fix_instructions: list[str]
    stage_complete: bool = False


class ReviewOrchestrator:
    """回环A 编排：生成 → 审核 → 注入 fix_instruction 重写，最多 max_attempts 轮。"""

    def __init__(
        self,
        responder: Responder,
        reviewer: Reviewer,
        max_attempts: int = 3,
        fallback_reply: str = DEFAULT_FALLBACK,
    ) -> None:
        self._responder = responder
        self._reviewer = reviewer
        self._max_attempts = max_attempts
        self._fallback_reply = fallback_reply

    def run(
        self,
        turn: TurnInstruction,
        history: list[Message],
        user_message: str,
        build_ctx: Callable[[str], L5Context],
        validate: Callable[[str], list[str]] | None = None,
        initial_fix: str | None = None,
    ) -> ReviewOutcome:
        fix: str | None = initial_fix
        history_reviews: list[ReviewResult] = []
        fix_instructions: list[str] = []
        last_stage_complete = False

        for _ in range(self._max_attempts):
            gen = self._responder.generate(
                turn, history, user_message, fix_instruction=fix
            )
            violations = validate(gen.reply) if validate else []
            if violations:
                result = ReviewResult(
                    harm="pass", boundary="pass", quality="fail",
                    fail_reason="L4本地校验未通过",
                    fix_instruction="请修正以下问题：" + "；".join(violations),
                )
            else:
                result = self._reviewer.review(build_ctx(gen.reply), history_reviews)
            history_reviews.append(result)
            if result.fix_instruction:
                fix_instructions.append(result.fix_instruction)
            last_stage_complete = gen.stage_complete
            if result.passed:
                return ReviewOutcome(
                    reply=gen.reply,
                    review_history=history_reviews,
                    attempts=len(history_reviews),
                    fallback_used=False,
                    fix_instructions=fix_instructions,
                    stage_complete=last_stage_complete,
                )
            fix = result.fix_instruction

        return ReviewOutcome(
            reply=self._fallback_reply,
            review_history=history_reviews,
            attempts=len(history_reviews),
            fallback_used=True,
            fix_instructions=fix_instructions,
            stage_complete=False,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_review_loop.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/review tests/test_review_loop.py
git commit -m "feat: 审核端口、Mock审核器与回环A编排（≤3轮重写+兜底）"
```

---

### Task 7: L4 门面服务（service）与 V4 自校验

**Files:**
- Create: `l4/service.py`
- Test: `tests/test_service.py`
- Test: `tests/test_l5_validation.py`

**Interfaces:**
- Consumes: `Settings`（Task 1）、所有 schemas（Task 2）、`get_backend`（Task 3）、`SessionStore/SessionData`（Task 4）、`Responder/GenerationResult`、`build_l5_context/build_evidence_refs/build_summary`（Task 5）、`ReviewOrchestrator/ReviewOutcome`、`MockReviewer`、`Reviewer`（Task 6）
- Produces:
  - `SessionRequest(session_id: str|None=None, role: RoleInstruction, report: DiagnosticReport, boundary: BoundaryInstruction, safety: SafetyPassport=SafetyPassport())`
  - `TurnRequest(session_id: str, user_message: str, turn: TurnInstruction)`
  - `TurnOutcome(session_id: str, reply: str, l4_substage: Substage, l5_context: L5Context, summary: StructuredSummary, review_history: list[ReviewResult], fallback_used: bool=False)`
  - `L4Service(config: Settings|None=None, backend: LLMBackend|None=None, reviewer: Reviewer|None=None)`：
    - `create_session(req: SessionRequest) -> str`：session_id 缺省自动生成（uuid4 hex）；anchor = `req.report.to_anchor()`；therapy_options = `[primary] + addon`；存入 SessionStore
    - `handle_turn(req: TurnRequest, initial_fix: str|None=None) -> TurnOutcome`：resolve substage → 回环A → V4 本地自校验（`_validate`）→ 构建 L5Context/StructuredSummary → 成功后推进状态机并写历史；fallback 时 flags=`loop_A_fallback`+`manual_review_required`
    - `_validate(reply, turn, therapy_options) -> list[str]`：禁止词检查 + V4-3 疗法一致性（回复提及 `THERAPY_KEYWORDS` 中不在 options 且非 CBT 的疗法词即违规）
  - `THERAPY_KEYWORDS: dict[str, str]`（service.py 模块级）

- [ ] **Step 1: 写失败测试**

`tests/test_service.py`：

```python
import json
from pathlib import Path

import pytest

from l4.config import Settings
from l4.llm.mock import MockBackend
from l4.review.mock_reviewer import MockReviewer
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    SafetyPassport,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.service import L4Service, SessionRequest, TurnRequest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

ROLE = RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化")
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来"],
                               safety_trigger="用户表达自杀意图→立即触发安全协议")
SAFETY = SafetyPassport(risk_level=1, sensitive_topics=[], flags=["monitor_suicidal_ideation"])


def _service(**backend_kwargs) -> L4Service:
    backend = MockBackend(**backend_kwargs)
    return L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())


def _report() -> DiagnosticReport:
    raw = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    return DiagnosticReport.model_validate(raw)


def _new_session(service: L4Service, session_id: str | None = None) -> str:
    return service.create_session(
        SessionRequest(session_id=session_id, role=ROLE, report=_report(),
                       boundary=BOUNDARY, safety=SAFETY)
    )


def test_create_session_auto_id_and_explicit_id():
    service = _service()
    sid = _new_session(service)
    assert len(sid) == 32
    sid2 = _new_session(service, "my-session")
    assert sid2 == "my-session"


def test_turn_end_to_end():
    service = _service()
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="我最近工作压力很大，晚上总是睡不好。",
        turn=TurnInstruction(goal="建立共情，确认当前状态", technique="开放式提问+反映",
                             force_substage="L4-L1"),
    ))
    assert out.reply
    assert out.l4_substage == "L4-L1"
    assert out.fallback_used is False
    assert out.l5_context.l4_raw_output == out.reply
    assert out.l5_context.l4_substage == "L4-L1"
    assert out.l5_context.diagnosis_context.suicidal_ideation == "passive_only"
    assert out.summary.layer_id == "L4"
    assert out.summary.evidence_refs  # 非空
    assert out.summary.flags == []


def test_turn_unknown_session_raises():
    service = _service()
    with pytest.raises(KeyError):
        service.handle_turn(TurnRequest(
            session_id="nope", user_message="hi",
            turn=TurnInstruction(goal="g", technique="t"),
        ))


def test_forbidden_word_triggers_rewrite():
    backend = MockBackend(
        responses={
            "请修正": '{"reply": "那我们先把目标放在睡个好觉上，你觉得呢？", "stage_complete": false, "reason": ""}'
        },
        default='{"reply": "你的情况属于确诊的疾病，建议你换个工作。", "stage_complete": false, "reason": ""}',
    )
    service = L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="最近压力很大",
        turn=TurnInstruction(goal="建立共情", technique="反映",
                             forbidden=["建议", "确诊"]),
    ))
    assert out.fallback_used is False
    assert "建议" not in out.reply and "确诊" not in out.reply
    assert "本地校验未通过" in out.review_history[0].fail_reason


def test_fallback_marks_flags():
    backend = MockBackend(default='{"reply": "你得了抑郁，建议开药。", "stage_complete": false, "reason": ""}')
    service = L4Service(Settings(_env_file=None), backend=backend,
                        reviewer=MockReviewer(always_fail=True))
    sid = _new_session(service)
    out = service.handle_turn(TurnRequest(
        session_id=sid,
        user_message="我感觉很糟",
        turn=TurnInstruction(goal="g", technique="t"),
    ))
    assert out.fallback_used is True
    assert "loop_A_fallback" in out.summary.flags
    assert "manual_review_required" in out.summary.flags
    assert out.reply == Settings(_env_file=None).fallback_reply


def test_handle_turn_initial_fix():
    backend = MockBackend(
        responses={"删除确定性措辞": '{"reply": "初步印象是情绪困扰，你感觉呢？", "stage_complete": false, "reason": ""}'},
        default='{"reply": "你得了中度抑郁。", "stage_complete": false, "reason": ""}',
    )
    service = L4Service(Settings(_env_file=None), backend=backend, reviewer=MockReviewer())
    sid = _new_session(service)
    out = service.handle_turn(
        TurnRequest(session_id=sid, user_message="hi",
                    turn=TurnInstruction(goal="g", technique="t")),
        initial_fix="删除确定性措辞，改为初步印象",
    )
    assert "初步印象" in out.reply
```

`tests/test_l5_validation.py`：

```python
from l4.service import L4Service, THERAPY_KEYWORDS


def test_validate_forbidden_words():
    service = L4Service()
    v = service._validate("你最好建议他辞职，我确诊了你的问题",
                          _turn(forbidden=["建议", "确诊"]), ["CBT"])
    assert any("建议" in x or "确诊" in x for x in v)


def test_validate_therapy_consistency_v43():
    service = L4Service()
    # MDD + CBT 方案下，回复提及 EMDR 属越界
    v = service._validate("我们可以尝试EMDR疗法", _turn(), ["CBT", "CBT-I"])
    assert any("EMDR" in x for x in v)
    # 允许的疗法不违规
    v2 = service._validate("我们试试CBT-I的睡眠限制", _turn(), ["CBT", "CBT-I"])
    assert v2 == []


def test_therapy_keywords_covered():
    assert "EMDR" in THERAPY_KEYWORDS
    assert "CBT-I" in THERAPY_KEYWORDS


def _turn(**kw):
    from l4.schemas.instructions import TurnInstruction
    d = dict(goal="g", technique="t")
    d.update(kw)
    return TurnInstruction(**d)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_service.py tests/test_l5_validation.py -v`
Expected: FAIL（ModuleNotFoundError: l4.service）

- [ ] **Step 3: 最小实现**

`l4/service.py`：

```python
import uuid

from pydantic import BaseModel

from l4.config import Settings
from l4.generation.responder import Responder
from l4.generation.summary_builder import (
    build_evidence_refs,
    build_l5_context,
    build_summary,
)
from l4.llm.base import LLMBackend
from l4.llm import get_backend
from l4.review.mock_reviewer import MockReviewer
from l4.review.orchestrator import ReviewOrchestrator
from l4.review.reviewer import Reviewer
from l4.schemas.instructions import (
    BoundaryInstruction,
    DiagnosticAnchor,
    RoleInstruction,
    SafetyPassport,
    Substage,
    TurnInstruction,
)
from l4.schemas.l3_report import DiagnosticReport
from l4.schemas.l5_context import L5Context
from l4.schemas.review import ReviewResult
from l4.schemas.summary import StructuredSummary
from l4.session.history import ConversationTurn, SessionData, SessionStore

THERAPY_KEYWORDS = {
    "EMDR": "EMDR", "CPT": "CPT", "IPSRT": "IPSRT", "ERP": "ERP",
    "CBT-I": "CBT-I", "ACT": "ACT", "MBCT": "MBCT", "IPT": "IPT",
    "SFBT": "SFBT", "CBT": "CBT",
}


class SessionRequest(BaseModel):
    session_id: str | None = None
    role: RoleInstruction
    report: DiagnosticReport
    boundary: BoundaryInstruction = BoundaryInstruction()
    safety: SafetyPassport = SafetyPassport()


class TurnRequest(BaseModel):
    session_id: str
    user_message: str
    turn: TurnInstruction


class TurnOutcome(BaseModel):
    session_id: str
    reply: str
    l4_substage: Substage
    l5_context: L5Context
    summary: StructuredSummary
    review_history: list[ReviewResult]
    fallback_used: bool = False


class L4Service:
    """L4 治疗对话层门面：一轮完整处理（输入指令 → 回复 + L5Context + 摘要）。"""

    def __init__(
        self,
        config: Settings | None = None,
        backend: LLMBackend | None = None,
        reviewer: Reviewer | None = None,
    ) -> None:
        self.config = config or Settings()
        self.backend = backend or get_backend(self.config)
        if self.config.review_mode == "l5_endpoint":
            raise NotImplementedError(
                "review_mode='l5_endpoint' 尚未实现，请使用 'mock'，或真实 L5 就绪后注入 reviewer 实现"
            )
        self.reviewer = reviewer or MockReviewer()
        self.store = SessionStore()

    def create_session(self, req: SessionRequest) -> str:
        session_id = req.session_id or uuid.uuid4().hex
        anchor = req.report.to_anchor()
        therapy_options = [t for t in [req.report.recommended_therapy.primary]
                           + list(req.report.recommended_therapy.addon) if t]
        data = SessionData(
            session_id=session_id,
            role=req.role,
            anchor=anchor,
            boundary=req.boundary,
            safety=req.safety,
            therapy_options=therapy_options,
        )
        self.store.create(data)
        return session_id

    def handle_turn(self, req: TurnRequest, initial_fix: str | None = None) -> TurnOutcome:
        sd = self.store.get(req.session_id)
        anchor = sd.anchor or DiagnosticAnchor()
        substage = sd.machine.resolve(req.turn.force_substage)

        responder = Responder(self.backend, sd.role, anchor, sd.boundary)
        orchestrator = ReviewOrchestrator(
            responder, self.reviewer,
            self.config.max_rewrite_attempts, self.config.fallback_reply,
        )

        def build_ctx(reply: str) -> L5Context:
            return build_l5_context(reply, substage, anchor.therapy_type, sd.safety, anchor)

        def validate(reply: str) -> list[str]:
            return self._validate(reply, req.turn, sd.therapy_options)

        outcome = orchestrator.run(
            req.turn, sd.history, req.user_message,
            build_ctx, validate, initial_fix=initial_fix,
        )

        flags: list[str] = list(sd.machine.flags)
        if outcome.fallback_used:
            flags.extend(["loop_A_fallback", "manual_review_required"])

        summary = build_summary(
            req.turn, substage, outcome.reply,
            build_evidence_refs(req.turn, sd.history, sd.safety),
            flags=flags,
        )

        sd.history.append(ConversationTurn(role="user", content=req.user_message))
        sd.history.append(ConversationTurn(role="assistant", content=outcome.reply))
        sd.l3_history.append(req.turn)
        if not outcome.fallback_used:
            sd.machine.advance(stage_complete=outcome.stage_complete,
                               force=req.turn.force_substage)

        return TurnOutcome(
            session_id=req.session_id,
            reply=outcome.reply,
            l4_substage=substage,
            l5_context=build_ctx(outcome.reply),
            summary=summary,
            review_history=outcome.review_history,
            fallback_used=outcome.fallback_used,
        )

    @staticmethod
    def _validate(reply: str, turn: TurnInstruction,
                  therapy_options: list[str]) -> list[str]:
        """V4 本地自校验：禁止词 + V4-3 疗法一致性（启发式）。"""
        violations: list[str] = []
        for word in turn.forbidden:
            if word and word in reply:
                violations.append(f"回复包含禁止词：{word}")
        allowed = set(therapy_options)
        for kw in THERAPY_KEYWORDS:
            if kw in reply and kw not in allowed and kw != "CBT":
                violations.append(f"回复提及未授权疗法：{kw}")
        return violations
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_service.py tests/test_l5_validation.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/service.py tests/test_service.py tests/test_l5_validation.py
git commit -m "feat: L4 门面服务（会话/回合处理、V4 本地自校验、兜底标记）"
```

---

### Task 8: FastAPI 服务

**Files:**
- Create: `l4/api/__init__.py`
- Create: `l4/api/server.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`（Task 1）、`L4Service/SessionRequest/TurnRequest/TurnOutcome`（Task 7）、`TurnInstruction`（Task 2）
- Produces:
  - `create_app(settings: Settings | None = None) -> FastAPI`：实例化 L4Service 并挂载端点
  - `SessionCreated(session_id: str)`（server.py）
  - `RewriteRequest(session_id: str, user_message: str, turn: TurnInstruction, fix_instruction: str)`（server.py）
  - 端点：`GET /healthz` → `{"status": "ok"}`；`POST /v1/l4/session`（body: SessionRequest，返回 SessionCreated）；`POST /v1/l4/turn`（body: TurnRequest，返回 TurnOutcome）；`POST /v1/l4/rewrite`（body: RewriteRequest，返回 TurnOutcome，外部回环A 入口）；会话不存在返回 HTTP 404
  - 模块级 `app = create_app()` 供 uvicorn 直接使用

- [ ] **Step 1: 写失败测试**

`tests/test_api.py`：

```python
import json
from pathlib import Path

from fastapi.testclient import TestClient

from l4.api.server import create_app
from l4.schemas.instructions import (
    BoundaryInstruction,
    RoleInstruction,
    TurnInstruction,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

ROLE = RoleInstruction(identity="CBT治疗师", style="温暖")
BOUNDARY = BoundaryInstruction(absolute_bans=["禁止开药"])


def _client():
    return TestClient(create_app())


def _session_payload(session_id=None):
    report = json.loads((EXAMPLES / "sample_diagnostic_report.json").read_text(encoding="utf-8"))
    payload = {"role": ROLE.model_dump(), "report": report,
               "boundary": BOUNDARY.model_dump()}
    if session_id:
        payload["session_id"] = session_id
    return payload


def test_healthz():
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_session_and_turn_flow():
    client = _client()
    resp = client.post("/v1/l4/session", json=_session_payload("s-api"))
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "s-api"

    turn = TurnInstruction(goal="建立共情", technique="开放式提问",
                           force_substage="L4-L1")
    resp = client.post("/v1/l4/turn", json={
        "session_id": "s-api",
        "user_message": "我最近压力很大，睡不好。",
        "turn": turn.model_dump(),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["l4_substage"] == "L4-L1"
    assert body["reply"]
    assert body["l5_context"]["l4_substage"] == "L4-L1"
    assert body["summary"]["layer_id"] == "L4"


def test_turn_unknown_session_404():
    resp = _client().post("/v1/l4/turn", json={
        "session_id": "nope",
        "user_message": "hi",
        "turn": TurnInstruction(goal="g", technique="t").model_dump(),
    })
    assert resp.status_code == 404


def test_rewrite_endpoint():
    client = _client()
    client.post("/v1/l4/session", json=_session_payload("s-rw"))
    resp = client.post("/v1/l4/rewrite", json={
        "session_id": "s-rw",
        "user_message": "hi",
        "turn": TurnInstruction(goal="g", technique="t").model_dump(),
        "fix_instruction": "删除确定性措辞，改为初步印象",
    })
    assert resp.status_code == 200
    assert resp.json()["reply"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL（ModuleNotFoundError: l4.api）

- [ ] **Step 3: 最小实现**

`l4/api/__init__.py`：

```python
from l4.api.server import app, create_app

__all__ = ["app", "create_app"]
```

`l4/api/server.py`：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from l4.config import Settings
from l4.schemas.instructions import TurnInstruction
from l4.service import L4Service, SessionRequest, TurnOutcome, TurnRequest


class SessionCreated(BaseModel):
    session_id: str


class RewriteRequest(BaseModel):
    session_id: str
    user_message: str
    turn: TurnInstruction
    fix_instruction: str


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="L4 治疗对话层", version="0.1.0")
    service = L4Service(settings or Settings())

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/l4/session", response_model=SessionCreated)
    def create_session(req: SessionRequest) -> SessionCreated:
        return SessionCreated(session_id=service.create_session(req))

    @app.post("/v1/l4/turn", response_model=TurnOutcome)
    def handle_turn(req: TurnRequest) -> TurnOutcome:
        try:
            return service.handle_turn(req)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"会话不存在: {req.session_id}")

    @app.post("/v1/l4/rewrite", response_model=TurnOutcome)
    def rewrite(req: RewriteRequest) -> TurnOutcome:
        """外部回环A 入口：接收 L5 的 fix_instruction 触发重写（备用，L4 内编排为主）。"""
        turn_req = TurnRequest(
            session_id=req.session_id,
            user_message=req.user_message,
            turn=req.turn,
        )
        try:
            return service.handle_turn(turn_req, initial_fix=req.fix_instruction)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"会话不存在: {req.session_id}")

    return app


app = create_app()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/api tests/test_api.py
git commit -m "feat: FastAPI 服务（会话/回合/重写端点 + 健康检查）"
```

---

### Task 9: CLI 演示与 README 补全

**Files:**
- Create: `l4/cli/__init__.py`
- Create: `l4/cli/demo.py`
- Modify: `README.md`（补全使用说明）
- Test: 无（演示脚本；用 `python -m l4.cli.demo --scripted` 验证）

**Interfaces:**
- Consumes: `L4Service/SessionRequest/TurnRequest`（Task 7）、`RoleInstruction/BoundaryInstruction/SafetyPassport/TurnInstruction`（Task 2）、`examples/sample_diagnostic_report.json`
- Produces: 可执行入口 `python -m l4.cli.demo`（`--scripted` 自动演示；无参数时交互式）

- [ ] **Step 1: 实现演示脚本**

`l4/cli/__init__.py`：

```python
from l4.cli.demo import main

__all__ = ["main"]
```

`l4/cli/demo.py`：

```python
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
```

- [ ] **Step 2: 运行演示验证**

Run: `python -m l4.cli.demo --scripted`
Expected: 输出 4 轮对话，每轮显示子阶段 L4-L1→L4-L4、回复、审核历史；无异常

- [ ] **Step 3: 补全 README**

`README.md` 全文：

```markdown
# Layer 4 治疗对话层（Treatment Dialogue Layer）

五层认知架构（安全评估 → MI评估 → 诊断推理 → **治疗对话** → 输出安全审核）的第四层。
本层接收 L3 诊断层下发的 L1-L4 治疗指令，将其"翻译"为自然、温暖、专业的治疗对话，
不做任何临床推理（临床推理与语言生成分离）。

## 模块结构

```
l4/
├── config.py         # 配置（环境变量前缀 L4_）
├── schemas/          # 层间数据契约（L1-L4 指令 / DiagnosticReport / L5Context / 摘要 / 审核）
├── llm/              # LLM 端口与适配器（OpenAI 兼容 / Mock / 本地模型占位）
├── session/          # L4-L1→L4-L4 子层状态机与会话历史
├── generation/       # prompt 四段式组装、回复生成、结构化摘要
├── review/           # 审核端口、Mock 审核器、回环A（≤3 轮重写+兜底）
├── service.py        # L4 门面：一轮完整处理
├── api/server.py     # FastAPI 服务
└── cli/demo.py       # CLI 演示
```

## 快速开始（无需 API 密钥）

```bash
pip install -e ".[dev]"
pytest                       # 跑全部单测（Mock 实现，无网络）
python -m l4.cli.demo --scripted   # 脚本化演示四阶段对话
python -m l4.cli.demo              # 交互式演示
uvicorn l4.api.server:app --port 8000   # 启动服务
```

## 接入真实 LLM（DeepSeek）

```bash
set L4_LLM_PROVIDER=openai_compat
set L4_LLM_BASE_URL=https://api.deepseek.com
set L4_LLM_API_KEY=sk-xxxx
set L4_LLM_MODEL=deepseek-chat
set L4_REVIEW_MODE=mock        # 真实 L5 就绪后替换为 l5_endpoint 或注入 reviewer
uvicorn l4.api.server:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/v1/l4/session` | 创建会话（role + DiagnosticReport + boundary + safety） |
| POST | `/v1/l4/turn` | 一轮处理（session_id + L3 指令 + 用户消息） |
| POST | `/v1/l4/rewrite` | 外部回环A 入口（携带 fix_instruction 重写） |

## 接口契约

- 输入：接口三 `DiagnosticReport`（L3→L4），见 `examples/sample_diagnostic_report.json`
- 输出：接口四 `ContextualSafetyContext`（L4+L3→L5）+ 结构化临床摘要（`StructuredSummary`，含 `evidence_refs` 证据链）
- 回环A：L5 fail → 注入 `fix_instruction` 重写，最多 3 轮，超限输出安全兜底回复并标记 `loop_A_fallback` + `manual_review_required`

## 接入本地训练模型

本地模型训练完成后，实现 `l4/llm/local.py`（vLLM/transformers），保持 `LLMBackend.chat()` 签名不变，
并设置 `L4_LLM_PROVIDER=local`，L4 其余代码零改动。

## 测试

```bash
pytest -v
```

单测全部使用 MockBackend + MockReviewer，无网络依赖。
```

- [ ] **Step 4: 全量回归**

Run: `python -m pytest -v`
Expected: 全部 PASS（约 42 passed）

- [ ] **Step 5: 提交**

```bash
git add l4/cli README.md
git commit -m "feat: CLI 演示脚本与 README 使用说明"
```

---

## 自检记录

**1. Spec 覆盖：**
- 输入契约（L1/L2/L3/L4 + DiagnosticReport）→ Task 2
- 输出契约（L5Context + StructuredSummary + ReviewResult）→ Task 2
- LLM 可插拔（OpenAI 兼容 + Mock + 本地占位）→ Task 3
- 完整状态机 L4-L1→L4-L4（L3 覆盖 + 回退标记 + 终态）→ Task 4
- 四段式 prompt 组装 + 插件嵌入 + fix_instruction 注入 → Task 5
- 回环A（≤3 轮 + 兜底 + 失败原因追溯）→ Task 6
- V4-1/V4-3 本地自校验 + V4-4 兜底标记 → Task 7
- FastAPI 服务（session/turn/rewrite/healthz）→ Task 8
- CLI 演示 + README → Task 9
- 层内自查指标（CBT 步骤完整性）→ Task 5 test_prompt_assembler 系列断言 + Task 9 演示四阶段

**2. 占位符扫描：** LocalModelBackend 的 NotImplementedError 为有意设计（本地模型占位），其余无 TBD/TODO。

**3. 类型一致性：** `get_backend(config)`、`SessionData`、`ReviewOutcome`、`TurnOutcome` 等签名在 Task 3-8 间交叉核对一致；`force_substage` 类型均为 `Substage | None`；`build_l5_context` 参数顺序在 Task 5 定义、Task 6/7 调用一致。
