"""Patient Agent：模拟稳定一致的患者，自动与完整系统跑完多轮测试。

模块分工（对应任务说明第四节的 PA1–PA7）：

    schemas.py    PA1  PatientScenario 病例 Schema
    library.py    PA2  版本化病例库
    generator.py  PA3  患者生成器（LLM 适配 + 确定性兜底）
    guard.py      PA4  一致性守卫
    runner.py     PA5  场景运行器
    stopping.py   PA6  停止条件
    evaluator.py  PA7  评测器
    report.py     PA7  结构化报告

    config.py     运行配置（环境变量 / .env）
    lexicon.py    与病例无关的共享词表
    errors.py     稳定错误码
    recorder.py   逐轮记录
    client/       完整系统 API 契约 + Mock 五层桩 + HTTP 实现
    llm/          LLM 适配器（离线伪模型 / OpenAI 兼容）
"""

from __future__ import annotations

from patient_agent.errors import (
    ErrorCode,
    LLMError,
    PatientAgentError,
    ScenarioLibraryError,
    ScenarioValidationError,
    SystemApiError,
)

__version__ = "0.1.0"

__all__ = [
    "ErrorCode",
    "LLMError",
    "PatientAgentError",
    "ScenarioLibraryError",
    "ScenarioValidationError",
    "SystemApiError",
    "__version__",
]
