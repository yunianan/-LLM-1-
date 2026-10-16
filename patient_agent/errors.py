"""稳定错误码与异常基类。

前端工作包要求区分：API 错误、超时、Schema 错误、知识检索错误、人工复核、会话停止
（见《心理诊疗系统五项后续开发任务说明》第五节 FE6）。Patient Agent 侧的异常一律带上
这里的错误码，便于自动测试统计错误类型。
"""

from __future__ import annotations

from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 10】错误码对齐（跨工作包共享词表）
# 接入什么：这里的 ErrorCode 是与前端 / 编排器共享的稳定错误码（任务说明第五节 FE6）。
# 何时需要改：其它工作包新增或改名错误码时，两边要同步；Patient Agent 抛出的所有异常
#           都带这里的码，自动测试据此统计错误类型。
# 注意：SAFETY_HOLD / HUMAN_REVIEW 属于正常转介路径而非故障，运行器不当作异常处理
#       （判定逻辑见 client/base.py 的 NON_FATAL_CODES）。
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCode(str, Enum):
    """对外稳定错误码。患者端只看到友好文案，开发者端看错误码。"""

    API_ERROR = "API_ERROR"
    TIMEOUT = "TIMEOUT"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    KNOWLEDGE_RETRIEVAL_ERROR = "KNOWLEDGE_RETRIEVAL_ERROR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SESSION_STOPPED = "SESSION_STOPPED"
    SAFETY_HOLD = "SAFETY_HOLD"


class PatientAgentError(Exception):
    """Patient Agent 所有异常的基类。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: ErrorCode = ErrorCode.API_ERROR,
        detail: object = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"[{self.error_code.value}] {self.message}"


class ScenarioValidationError(PatientAgentError):
    """病例文件不符合 PatientScenario Schema。"""

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message, error_code=ErrorCode.SCHEMA_ERROR, detail=detail)


class SystemApiError(PatientAgentError):
    """调用完整系统 API（顶层编排器）失败。"""


class LLMError(PatientAgentError):
    """患者生成器调用 LLM 失败。"""


class ScenarioLibraryError(PatientAgentError):
    """病例库索引或目录结构有问题。"""

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message, error_code=ErrorCode.SCHEMA_ERROR, detail=detail)
