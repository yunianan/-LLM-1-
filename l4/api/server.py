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


def create_app(settings: Settings | None = None,
               service: L4Service | None = None) -> FastAPI:
    app = FastAPI(title="L4 治疗对话层", version="0.1.0")
    service = service or L4Service(settings or Settings())

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
            return service.handle_turn(turn_req, initial_fix=req.fix_instruction,
                                       persist=False)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"会话不存在: {req.session_id}")

    return app


app = create_app()
