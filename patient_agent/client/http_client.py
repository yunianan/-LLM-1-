"""完整系统 API 的真实 HTTP 客户端。

契约与 ``MockSystemApiClient`` 完全一致（同一个 ``SystemApiClient`` 抽象），
所以 Patient Agent 的运行器、守卫、评测器不需要知道对面是桩还是真编排器。

设计取舍：
- 只用标准库 ``urllib.request``，不引入 requests/httpx，避免给别的层增加依赖。
- 失败一律翻译成 ``SystemApiError``，并按语义带上 ``ErrorCode``：
  连不上/5xx→API_ERROR，超时→TIMEOUT，返回体不符合 Schema→SCHEMA_ERROR。
- 响应体交给 Pydantic 解析；``TurnResponse`` 等模型都是 ``extra="ignore"``，
  编排器加字段不会把测试打挂。
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 2】真实五层编排器 —— HTTP 实现（本文件就是「接上真系统」的地方）
# 接入什么：真实系统的 HTTP 服务。本文件把它翻译成 SystemApiClient 的 8 个方法。
# 谁提供：编排器 / 后端工作包。
# 怎么打开：SYSTEM_API_MODE=http + SYSTEM_API_BASE_URL（可选 SYSTEM_API_TOKEN）。
# 需要的 8 条路由（方法 路径 → 对应方法）：
#   POST /sessions                             创建会话    → create_session
#   POST /sessions/{id}/messages               发送消息    → send_message
#   POST /sessions/{id}/profile_confirmation   画像确认    → confirm_profile
#   POST /sessions/{id}/scale                  提交量表    → submit_scale
#   POST /sessions/{id}/pause                  暂停/恢复   → set_pause
#   GET  /sessions/{id}/dev_state              开发状态    → get_dev_state
#   GET  /sessions/{id}/episode_summary        疗程总结    → get_episode_summary
#   GET  /sessions?session_id=...              历史疗程    → get_session_history
# 若真实后端的路径或字段不同：只改本文件对应的几行，上层运行器/守卫/评测器一行不动。
# base_url 形如 http://host:port/api（是否带 /api 由后端决定，客户端只做字符串拼接）。
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import ValidationError

from patient_agent.client.base import (
    DevState,
    EpisodeSummary,
    ProfileDecision,
    SessionHistoryItem,
    SessionSnapshot,
    SystemApiClient,
    TurnResponse,
)
from patient_agent.errors import ErrorCode, SystemApiError


class HttpSystemApiClient(SystemApiClient):
    """通过 HTTP 调用顶层编排器。``base_url`` 形如 ``http://host:port/api``。"""

    name = "http_system_api"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        token: str = "",
        session_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token
        self.session_id = session_id

    # ------------------------------------------------------------ 传输层
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            params = urllib.parse.urlencode(
                {key: value for key, value in query.items() if value is not None}
            )
            if params:
                url = f"{url}?{params}"

        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except socket.timeout as exc:
            raise SystemApiError(
                f"调用 {path} 超时（{self.timeout}s）",
                error_code=ErrorCode.TIMEOUT,
                detail=str(exc),
            ) from exc
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - 读错误体失败时忽略
                body = ""
            raise SystemApiError(
                f"调用 {path} 失败：HTTP {exc.code}",
                error_code=ErrorCode.API_ERROR,
                detail={"status": exc.code, "body": body},
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise SystemApiError(
                    f"调用 {path} 超时（{self.timeout}s）",
                    error_code=ErrorCode.TIMEOUT,
                    detail=str(exc.reason),
                ) from exc
            raise SystemApiError(
                f"无法连接完整系统：{exc.reason}",
                error_code=ErrorCode.API_ERROR,
                detail=str(exc.reason),
            ) from exc

        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemApiError(
                f"调用 {path} 返回的响应不是合法 JSON",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail={"raw": raw[:500]},
            ) from exc
        if not isinstance(parsed, dict):
            raise SystemApiError(
                f"调用 {path} 的响应应为 JSON 对象，实际是 {type(parsed).__name__}",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail={"raw": raw[:500]},
            )
        return parsed

    @staticmethod
    def _parse(model: type[Any], payload: dict[str, Any], path: str) -> Any:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise SystemApiError(
                f"调用 {path} 的响应不符合 {model.__name__} Schema",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail=exc.errors(),
            ) from exc

    def _session_path(self, session_id: str, suffix: str = "") -> str:
        return f"/sessions/{session_id}{suffix}"

    # ------------------------------------------------------------ 会话
    def create_session(
        self, *, locale: str = "zh-CN", channel: str = "test", session_id: str | None = None
    ) -> SessionSnapshot:
        payload: dict[str, Any] = {"locale": locale, "channel": channel}
        if session_id is not None:
            payload["session_id"] = session_id
        body = self._request("POST", "/sessions", payload)
        snapshot = self._parse(SessionSnapshot, body, "/sessions")
        self.session_id = snapshot.session_id
        return snapshot

    def send_message(self, session_id: str, text: str) -> TurnResponse:
        body = self._request(
            "POST", self._session_path(session_id, "/messages"), {"text": text}
        )
        return self._parse(TurnResponse, body, "POST /messages")

    # ------------------------------------------------------ 画像 / 量表 / 暂停
    def confirm_profile(
        self,
        session_id: str,
        decision: ProfileDecision | str = ProfileDecision.CONTINUE,
        notes: str = "",
    ) -> TurnResponse:
        value = decision.value if isinstance(decision, ProfileDecision) else str(decision)
        body = self._request(
            "POST",
            self._session_path(session_id, "/profile_confirmation"),
            {"decision": value, "notes": notes},
        )
        return self._parse(TurnResponse, body, "POST /profile_confirmation")

    def submit_scale(
        self,
        session_id: str,
        item_number: str,
        score: int,
        answer_text: str = "",
    ) -> TurnResponse:
        body = self._request(
            "POST",
            self._session_path(session_id, "/scale"),
            {"item_number": item_number, "score": score, "answer_text": answer_text},
        )
        return self._parse(TurnResponse, body, "POST /scale")

    def set_pause(self, session_id: str, paused: bool) -> TurnResponse:
        body = self._request(
            "POST", self._session_path(session_id, "/pause"), {"paused": paused}
        )
        return self._parse(TurnResponse, body, "POST /pause")

    # ------------------------------------------------------------ 状态与总结
    def get_dev_state(self, session_id: str) -> DevState:
        body = self._request("GET", self._session_path(session_id, "/dev_state"))
        return self._parse(DevState, body, "GET /dev_state")

    def get_episode_summary(self, session_id: str) -> EpisodeSummary | None:
        body = self._request("GET", self._session_path(session_id, "/episode_summary"))
        if not body:
            return None
        return self._parse(EpisodeSummary, body, "GET /episode_summary")

    def get_session_history(self, session_id: str | None = None) -> list[SessionHistoryItem]:
        body = self._request("GET", "/sessions", query={"session_id": session_id})
        items = body.get("items", body)
        if not isinstance(items, list):
            raise SystemApiError(
                "GET /sessions 的响应缺少 items 列表",
                error_code=ErrorCode.SCHEMA_ERROR,
                detail={"raw": str(body)[:500]},
            )
        parsed: list[SessionHistoryItem] = []
        for item in items:
            parsed.append(self._parse(SessionHistoryItem, item, "GET /sessions"))
        return parsed
