"""完整系统 API 客户端：抽象契约 + 内置 Mock 五层桩 + 真实 HTTP 实现。

运行时选择哪个实现由 ``config.Settings`` 决定；上层（运行器/守卫/评测器）
只依赖 ``SystemApiClient`` 抽象，因此离线跑测试与连真编排器是同一套代码路径。
"""

from __future__ import annotations

from patient_agent.client.base import (
    DevState,
    EpisodeSummary,
    ProfileDecision,
    RouteInfo,
    ScaleItemPrompt,
    SessionHistoryItem,
    SessionSnapshot,
    SystemApiClient,
    TurnResponse,
    new_session_id,
    raise_for_error,
)
from patient_agent.client.mock_system import MockConfig, MockSystemApiClient

__all__ = [
    "DevState",
    "EpisodeSummary",
    "MockConfig",
    "MockSystemApiClient",
    "ProfileDecision",
    "RouteInfo",
    "ScaleItemPrompt",
    "SessionHistoryItem",
    "SessionSnapshot",
    "SystemApiClient",
    "TurnResponse",
    "new_session_id",
    "raise_for_error",
]


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 3】客户端选择开关（内置五层桩 ←→ 真实编排器）
# 接入什么：运行期决定用 MockSystemApiClient 还是 HttpSystemApiClient。
# 怎么打开：config.Settings.system_api_mode（环境变量 SYSTEM_API_MODE=mock|http）。
#           默认 mock，离线可跑；改成 http 即走真实系统（还需 SYSTEM_API_BASE_URL）。
# 说明：真实实现延迟导入，offline 跑测试不会碰 urllib 细节。
# ─────────────────────────────────────────────────────────────────────────────


def load_http_client(*args: object, **kwargs: object) -> SystemApiClient:
    """延迟导入 HTTP 客户端。

    ``http_client`` 只在真的要走网络时才 import，离线跑测试不必碰 urllib 细节。
    """

    from patient_agent.client.http_client import HttpSystemApiClient

    return HttpSystemApiClient(*args, **kwargs)  # type: ignore[arg-type]
