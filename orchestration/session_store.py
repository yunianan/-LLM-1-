from copy import deepcopy

from orchestration.schemas import PipelineSession


class PipelineSessionStore:
    """顶层编排器的内存会话存储；后续可替换为 Redis/数据库实现。"""

    def __init__(self) -> None:
        self._sessions: dict[str, PipelineSession] = {}

    def create(self, session: PipelineSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"编排会话已存在：{session.session_id}")
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> PipelineSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"编排会话不存在：{session_id}") from exc

    def snapshot(self, session_id: str) -> PipelineSession:
        return deepcopy(self.get(session_id))
