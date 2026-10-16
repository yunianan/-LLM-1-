"""PA5 场景运行器：把「生成器 + 守卫 + 完整系统」串成一个自动多轮会话。

一次 ``run()`` 就是：建会话 → 患者说开场 → 循环「生成动作 → 守卫检查 → 调对应 API
→ 记录 → 判断是否停止」→ 收尾时拉开发状态、疗程总结与历史。

运行器负责调用系统 API；真实模型模式的生成与事实复核会调用模型 API。
"""

from __future__ import annotations

import time
from uuid import uuid4
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from patient_agent.client.base import (
    ProfileDecision,
    RouteInfo,
    ScaleItemPrompt,
    SystemApiClient,
    TurnResponse,
    raise_for_error,
)
from patient_agent.errors import ErrorCode, PatientAgentError, SystemApiError
from patient_agent.generator import PatientAction, PatientGenerator, TurnContext
from patient_agent.guard import ConsistencyGuard, GuardResult
from patient_agent.recorder import SessionRecorder, GenerationAttempt
from patient_agent.schemas import PatientScenario
from patient_agent.stopping import evaluate_stop


@dataclass
class RunOptions:
    """运行期的开关，用于 CLI 与测试微调行为。"""

    # 守卫发现阻断性违规时，最多重说几次（在线模式下有意义）。
    max_retries: int = 2
    # 注入润色 prompt 的最近医患对话轮数；0 表示不注入，退回旧行为。
    history_turns: int = 4
    # 会话结束后是否额外拉取开发状态 / 总结 / 历史。
    collect_artifacts: bool = True
    # 出错（系统 API 抛异常）时是否吞掉并正常收尾，便于批量跑病例。
    tolerate_errors: bool = True


class SessionRunner:
    def __init__(
        self,
        scenario: PatientScenario,
        client: SystemApiClient,
        generator: PatientGenerator,
        guard: ConsistencyGuard,
        *,
        options: RunOptions | None = None,
        llm_mode: str = "offline",
        llm_name: str = "",
    ) -> None:
        self.scenario = scenario
        self.client = client
        self.generator = generator
        self.guard = guard
        self.options = options or RunOptions()
        self.llm_mode = llm_mode
        self.llm_name = llm_name
        self.last_attempts: list[GenerationAttempt] = []

    # ------------------------------------------------------------------ 主循环
    def run(self) -> SessionRecorder:
        scenario = self.scenario
        self.generator.reset()

        recorder = SessionRecorder(
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.version,
            scenario_title=scenario.title,
            seed=scenario.seed,
            llm_mode=self.llm_mode,
            llm_name=self.llm_name,
            system_api_name=getattr(self.client, "name", "system_api"),
        )

        snapshot = self.client.create_session(locale=scenario.locale)
        recorder.session_id = snapshot.session_id
        if snapshot.initial_prompt:
            recorder.note(f"初始提示：{snapshot.initial_prompt}")

        # 开场：患者先说话。
        opening = self.generator.opening()
        started = time.perf_counter()
        opening, opening_guard, opening_retries = self.validate_action(opening)
        if not opening_guard.ok:
            self._blocked(recorder, 0, opening, opening_guard, opening_retries)
            return recorder
        try:
            response = raise_for_error(
                self.client.send_message(snapshot.session_id, opening.text)
            )
        except SystemApiError as exc:
            self._record(recorder, 0, opening, opening.text, self._error_turn(snapshot.session_id, exc), opening_guard,
                retries=opening_retries, elapsed_ms=(time.perf_counter()-started)*1000)
            return self._handle_fatal(recorder, opening, exc)

        self._record(
            recorder,
            0,
            opening,
            self.generator.action_to_send_text(opening),
            response,
            opening_guard,
            retries=opening_retries,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._track_after_send(recorder, opening, response)

        turn_index = 1
        while True:
            decision = evaluate_stop(
                scenario.stop,
                turn_count=turn_index,
                response=response,
                recorder=recorder,
            )
            if decision.should_stop:
                recorder.stop_reason = decision.reason
                break
            if turn_index >= scenario.stop.max_turns + 2:
                # 双保险：停止判断本身出问题时不至于死循环。
                recorder.stop_reason = "安全上限触发"
                break

            context = self._build_context(recorder, response, turn_index)
            started = time.perf_counter()
            # 暂停计划：按脚本在指定轮次后发起暂停。
            if self._should_pause(recorder, turn_index):
                self.last_attempts = []
                pause_response = raise_for_error(
                    self.client.set_pause(recorder.session_id, True)
                )
                self._record(
                    recorder,
                    turn_index,
                    PatientAction(kind="set_pause", paused=True, reason="按计划主动暂停"),
                    "[暂停]",
                    pause_response,
                    GuardResult(ok=True),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    note="患者主动暂停",
                )
                turn_index += 1
                response = pause_response
                continue

            action = self.generator.next_action(context)
            if action.kind == "stop":
                recorder.stop_reason = action.reason or "生成器主动结束"
                break

            guard_result = GuardResult(ok=True)
            patient_text = self.generator.action_to_send_text(action)
            retries = 0
            try:
                action, guard_result, retries = self.validate_action(action, context)
                patient_text = self.generator.action_to_send_text(action)
                if not guard_result.ok:
                    self._blocked(recorder, turn_index, action, guard_result, retries)
                    break
                response = raise_for_error(self._dispatch(recorder.session_id, action))
            except SystemApiError as exc:
                if not self.options.tolerate_errors:
                    raise
                response = self._error_turn(recorder.session_id, exc)
                recorder.note(f"第 {turn_index} 轮系统 API 异常：{exc}")
                self._record(
                    recorder,
                    turn_index,
                    action,
                    patient_text,
                    response,
                    guard_result,
                    retries=retries,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
                recorder.stop_reason = f"系统 API 异常：{exc.error_code.value}"
                break

            self._record(
                recorder,
                turn_index,
                action,
                patient_text,
                response,
                guard_result,
                retries=retries,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            self._track_after_send(recorder, action, response)
            turn_index += 1

        if self.options.collect_artifacts:
            self._collect_artifacts(recorder)
        return recorder

    def validate_action(self, action: PatientAction, context: TurnContext | None = None):
        """所有入口共用：重试耗尽或复核不可用时绝不放行。"""
        retries = 0
        self.last_attempts = []
        while True:
            started = time.perf_counter()
            result = self.guard.check(action, probe_text=context.last_reply if context else "",
                scale_item=context.scale_item if context else None,
                events=context.events if context else [])
            self.last_attempts.append(GenerationAttempt(attempt_id=uuid4().hex,
                previous_attempt_id=self.last_attempts[-1].attempt_id if self.last_attempts else None,
                attempt_number=retries+1, doctor_question=context.last_reply if context else "",
                selection_reason=action.reason, selected_fact_id=action.fact_id,
                authorized_evidence=result.authorized_evidence,
                generation_evidence=action.authorized_evidence,
                observed_events=list(context.events) if context else [],
                candidate_text=self.generator.action_to_send_text(action), segments=action.segments,
                raw_output=action.raw_output, generation_error=action.generation_error,
                generated_at=action.generated_at, generation_elapsed_ms=action.generation_elapsed_ms,
                review_elapsed_ms=(time.perf_counter()-started)*1000, model_id=self.llm_name,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
                prompt_version=action.prompt_version, prompt_system=action.prompt_system,
                prompt_user=action.prompt_user,
                action_payload={key: getattr(action, key) for key in ("kind", "decision", "notes", "item_number", "score", "answer_text", "paused")},
                guard_result=result.model_copy(deep=True)))
            if result.ok or action.kind != "say" or retries >= self.options.max_retries or any(not v.retryable for v in result.blocking):
                return action, result, retries
            retries += 1
            action = self.generator.retry_action(action, feedback="；".join(v.code.value + ":" + v.message for v in result.blocking))

    def _blocked(self, recorder, turn_index, action, result, retries):
        from patient_agent.recorder import TurnRecord
        recorder.turns.append(TurnRecord(turn_index=turn_index, action_kind=action.kind,
            patient_fact_id=action.fact_id, patient_denial=action.is_denial,
            submitted=False, rejected_text=self.generator.action_to_send_text(action),
            main_layer="", guard_ok=False, retries=retries,
            violations=[v.model_dump(mode="json") for v in result.violations]))
        recorder.turns[-1].attempts = [a.model_copy(deep=True) for a in self.last_attempts]
        recorder.turns[-1].submission_status = "blocked"
        recorder.stop_reason = "患者守卫阻断：" + ", ".join(result.codes())

    # ------------------------------------------------------------------ 分发
    def _dispatch(self, session_id: str, action: PatientAction) -> TurnResponse:
        if action.kind == "say":
            return self.client.send_message(session_id, action.text)
        if action.kind == "confirm_profile":
            decision = action.decision or ProfileDecision.CONTINUE.value
            return self.client.confirm_profile(session_id, decision, action.notes)
        if action.kind == "submit_scale":
            return self.client.submit_scale(
                session_id, action.item_number, action.score, action.answer_text
            )
        if action.kind == "set_pause":
            return self.client.set_pause(session_id, action.paused)
        raise ValueError(f"无法分发的动作：{action.kind}")

    # ------------------------------------------------------------------ 上下文
    def _build_context(
        self, recorder: SessionRecorder, response: TurnResponse, turn_index: int
    ) -> TurnContext:
        recent = (
            recorder.turns[-self.options.history_turns :]
            if self.options.history_turns > 0
            else []
        )
        return TurnContext(
            turn_index=turn_index,
            treatment_turn=recorder.treatment_turns,
            last_reply=response.patient_reply,
            recent_turns=[(turn.system_reply, turn.patient_text) for turn in recent],
            waiting_type=response.waiting_type,
            main_layer=response.route.main_layer,
            scale_item=response.scale_item,
            events=recorder.all_events(),
            session_ended=response.session_ended,
            termination_reason=response.termination_reason,
        )

    def _track_after_send(
        self, recorder: SessionRecorder, action: PatientAction, response: TurnResponse
    ) -> None:
        """把「进入治疗层」计成治疗轮次，供轨迹步推进使用。"""

        if response.route.main_layer == "L4" and action.kind in ("say", "submit_scale", "confirm_profile"):
            recorder.treatment_turns += 1

    def _should_pause(self, recorder: SessionRecorder, turn_index: int) -> bool:
        plan = self.scenario.pause_plan
        if plan.pause_after_turn is None:
            return False
        if plan.pause_after_turn != turn_index:
            return False
        # 只暂停一次，且暂停后马上会由生成器恢复。
        return not self.generator.state.pause_used

    # ------------------------------------------------------------------ 记录
    def _record(
        self,
        recorder: SessionRecorder,
        turn_index: int,
        action: PatientAction,
        patient_text: str,
        response: TurnResponse,
        guard_result: GuardResult,
        retries: int = 0,
        elapsed_ms: float = 0.0,
        note: str = "",
    ) -> None:
        if action.kind == "set_pause" and action.paused:
            self.generator.state.pause_used = True
        if action.kind == "say" or action.kind == "submit_scale":
            self.generator.state.send_turns += 1
        recorder.add_turn(
            turn_index=turn_index,
            action=action,
            patient_text=patient_text,
            response=response,
            guard_ok=guard_result.ok,
            violations=guard_result.violations,
            retries=retries,
            elapsed_ms=elapsed_ms,
            note=note,
            attempts=self.last_attempts,
        )
        if recorder.turns[-1].attempts:
            recorder.turns[-1].attempts[-1].submission_status = recorder.turns[-1].submission_status
            recorder.turns[-1].attempts[-1].submitted_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ 收尾
    def _collect_artifacts(self, recorder: SessionRecorder) -> None:
        if not recorder.session_id:
            return
        try:
            state = self.client.get_dev_state(recorder.session_id)
            recorder.final_state = state.model_dump(mode="json")
        except PatientAgentError as exc:  # pragma: no cover - 桩一般不会失败
            recorder.note(f"读取开发状态失败：{exc}")
        try:
            summary = self.client.get_episode_summary(recorder.session_id)
            recorder.episode_summary = summary.model_dump(mode="json") if summary else None
        except PatientAgentError as exc:  # pragma: no cover
            recorder.note(f"读取疗程总结失败：{exc}")
        try:
            history = self.client.get_session_history(recorder.session_id)
            recorder.history = [item.model_dump(mode="json") for item in history]
        except PatientAgentError as exc:  # pragma: no cover
            recorder.note(f"读取历史疗程失败：{exc}")

    def _handle_fatal(
        self, recorder: SessionRecorder, action: PatientAction, exc: SystemApiError
    ) -> SessionRecorder:
        if not self.options.tolerate_errors:
            raise exc
        recorder.note(f"开场即失败：{exc}")
        recorder.stop_reason = f"系统 API 异常：{exc.error_code.value}"
        return recorder

    def _error_turn(self, session_id: str, exc: SystemApiError) -> TurnResponse:
        """把异常转成一个「带错误码的回复」，让报告能统计到错误类型。

        路由显式标成 L5/error：异常轮没有真实层参与，若沿用 RouteInfo 的默认
        ``main_layer="L1"``，会让图层覆盖统计平白多出一个 L1。
        """

        return TurnResponse(
            session_id=session_id,
            patient_reply="",
            waiting_type="none",
            route=RouteInfo(main_layer="L5", intent="error", reason=exc.message),
            error_code=exc.error_code or ErrorCode.API_ERROR,
            error_message=exc.message,
        )
