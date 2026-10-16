"""完整系统 API 契约（顶层编排器）。

接口清单来自任务说明第五节「完整系统API」，与前端用的是同一套：

    创建会话 / 发送消息 / 画像确认 / 提交量表 / 暂停恢复
    获取开发状态 / 获取疗程总结 / 获取历史疗程

Patient Agent 只通过这些接口与系统交互：不直接调用单层，不自己算风险、诊断、
量表分数或审核结果（第四节负责人交付边界）。

响应类模型统一 ``extra="ignore"``：编排器后续新增字段时不该把自动测试全部打挂；
请求侧仍然严格，写错字段会立刻报错。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from patient_agent.errors import ErrorCode, SystemApiError

WaitingType = Literal[
    "none",
    "profile_confirmation",
    "scale_item",
    "pause",
    "safety_hold",
    "human_review",
    "completed",
]
MainLayer = Literal["L1", "L2", "L3", "L4", "L5"]


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RouteInfo(ResponseModel):
    main_layer: MainLayer = "L1"
    secondary_layers: list[str] = Field(default_factory=list)
    intent: str = ""
    reason: str = ""


class ScaleItemPrompt(ResponseModel):
    scale_name: str
    item_number: str
    prompt: str
    min_score: int = 0
    max_score: int = 4


class SessionSnapshot(ResponseModel):
    session_id: str
    route: RouteInfo = Field(default_factory=RouteInfo)
    initial_prompt: str = ""
    locale: str = "zh-CN"


class TurnResponse(ResponseModel):
    session_id: str
    patient_reply: str
    waiting_type: WaitingType = "none"
    route: RouteInfo = Field(default_factory=RouteInfo)
    scale_item: ScaleItemPrompt | None = None
    events: list[str] = Field(default_factory=list)
    error_code: ErrorCode | None = None
    error_message: str = ""
    # 会话是否已被系统终止（危机转介、人工复核、疗程结束、主动退出）。
    session_ended: bool = False
    termination_reason: str | None = None


class DevState(ResponseModel):
    session_id: str
    turn_count: int = 0
    waiting_type: WaitingType = "none"
    route: RouteInfo = Field(default_factory=RouteInfo)
    layers: dict[str, Any] = Field(default_factory=dict)
    rag_hits: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: dict[str, list[str]] = Field(default_factory=dict)
    agent_status: dict[str, Any] = Field(default_factory=dict)


class EpisodeSummary(ResponseModel):
    episode_id: str
    session_id: str
    termination_reason: str
    turns: int = 0
    route_summary: dict[str, int] = Field(default_factory=dict)
    l2_intake: dict[str, Any] = Field(default_factory=dict)
    l3_result: dict[str, Any] = Field(default_factory=dict)
    l4_summary: dict[str, Any] = Field(default_factory=dict)
    l5_reviews: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chain: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SessionHistoryItem(ResponseModel):
    session_id: str
    episode_id: str | None = None
    created_at: str = ""
    termination_reason: str | None = None
    version: str = ""


class ProfileDecision(str, Enum):
    CONTINUE = "continue"
    MODIFY = "modify"
    PAUSE = "pause"


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 1】真实五层编排器 —— 契约层
# 接入什么：由真实系统（L1~L5 + 顶层编排器）实现的 8 个接口。真实后端只要在响应
#           形态上满足这里定义的 Pydantic 模型，Patient Agent 就能直接跑。
# 谁提供：编排器 / 后端工作包。
# 怎么打开：本文件不需要改。实现写在 client/http_client.py（接入点 2），
#           开关在 config.Settings.system_api_mode（接入点 3、9）。
# 对齐要点：
#   * 8 条 HTTP 路径见 client/http_client.py 的【接入点 2】；
#   * RouteInfo.main_layer、TurnResponse.waiting_type、termination_reason 是跨端
#     约定的路由词表（本文件第 26~35 行），任一方改词都要各方同步；
#   * 响应模型一律 extra="ignore"：编排器后续新增字段不会把 Patient Agent 的测试打挂。
# ─────────────────────────────────────────────────────────────────────────────


class SystemApiClient(ABC):
    """完整系统 API 客户端抽象。Mock 桩与真实 HTTP 客户端都实现它。"""

    name: str = "system_api"

    # ------------------------------------------------------------------ 会话
    @abstractmethod
    def create_session(
        self, *, locale: str = "zh-CN", channel: str = "test", session_id: str | None = None
    ) -> SessionSnapshot:
        """创建会话，返回 session_id、初始路由和患者可见的初始提示。"""

    @abstractmethod
    def send_message(self, session_id: str, text: str) -> TurnResponse:
        """把患者消息交给顶层编排器，返回患者回复、等待类型、路由和错误码。"""

    # ------------------------------------------------------- 画像 / 量表 / 暂停
    @abstractmethod
    def confirm_profile(
        self,
        session_id: str,
        decision: ProfileDecision | str = ProfileDecision.CONTINUE,
        notes: str = "",
    ) -> TurnResponse:
        """提交「继续 / 修改 / 暂停」的画像确认决定。"""

    @abstractmethod
    def submit_scale(
        self,
        session_id: str,
        item_number: str,
        score: int,
        answer_text: str = "",
    ) -> TurnResponse:
        """提交当前许可量表题目的答案；返回下一题或回到访谈。"""

    @abstractmethod
    def set_pause(self, session_id: str, paused: bool) -> TurnResponse:
        """控制患者主动暂停状态，返回当前可执行动作。"""

    # ------------------------------------------------------------ 状态与总结
    @abstractmethod
    def get_dev_state(self, session_id: str) -> DevState:
        """读取开发者可见的结构化流程快照。"""

    @abstractmethod
    def get_episode_summary(self, session_id: str) -> EpisodeSummary | None:
        """读取 L5 生成的周期总结；未结束时返回 None。"""

    @abstractmethod
    def get_session_history(self, session_id: str | None = None) -> list[SessionHistoryItem]:
        """读取外部 Memory 中的脱敏疗程列表。"""


def new_session_id(prefix: str = "S") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


#: 这些错误码代表「流程按设计走到了专门路径」，响应本身携带有效的患者可见回复，
#: 不属于异常：运行器应当记录该轮并由 PA6 以 system_terminated 停止。
NON_FATAL_CODES: frozenset[ErrorCode] = frozenset(
    {ErrorCode.SAFETY_HOLD, ErrorCode.HUMAN_REVIEW}
)


def raise_for_error(response: TurnResponse) -> TurnResponse:
    """把带错误码的响应转成异常，方便运行器集中处理。

    ``SAFETY_HOLD`` / ``HUMAN_REVIEW`` 是正常转介路径而非故障：它们照常返回
    patient_reply 与路由，运行器需要把这轮对话记下来再停；只有其它错误码才抛。
    """

    if response.error_code is not None and response.error_code not in NON_FATAL_CODES:
        raise SystemApiError(
            response.error_message or f"完整系统 API 返回错误 {response.error_code.value}",
            error_code=response.error_code,
            detail=response.model_dump(),
        )
    return response
