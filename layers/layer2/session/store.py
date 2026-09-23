from copy import deepcopy

from layers.layer2.schemas import L2SessionState


class L2SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, L2SessionState] = {}

    def create(self, state: L2SessionState) -> None:
        if state.session_id in self._sessions:
            raise ValueError(f"L2会话已存在：{state.session_id}")
        self._sessions[state.session_id] = state

    def get(self, session_id: str) -> L2SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"L2会话不存在：{session_id}") from exc

    def replace(self, state: L2SessionState) -> None:
        if state.session_id not in self._sessions:
            raise KeyError(f"L2会话不存在：{state.session_id}")
        self._sessions[state.session_id] = state

    def snapshot(self, session_id: str) -> L2SessionState:
        return deepcopy(self.get(session_id))

