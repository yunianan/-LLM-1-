from __future__ import annotations

from layers.layer2.profile import is_sufficient, missing_dimensions
from layers.layer2.scales import SCALE_CATALOG, ScaleEngine
from layers.layer2.schemas import L2SessionState, MIStage, SemanticAnalysis, StageDecision
from shared.l2_l3_contract import REQUIRED_DIAGNOSTIC_FIELD_BY_LABEL


class MIStageController:
    """规则状态机；不按固定轮数推进，也不把阶段名称暴露给患者。"""

    def advance(self, state: L2SessionState, analysis: SemanticAnalysis) -> None:
        if state.confirmation_status in {"paused", "revision_requested"}:
            return
        if state.current_stage == "engaging" and (
            state.symptoms or state.key_narratives
        ):
            state.current_stage = "focusing"
            return
        if state.current_stage == "focusing" and self._focusing_complete(state):
            state.current_stage = "evoking"
            return
        if state.current_stage == "evoking" and is_sufficient(state):
            state.current_stage = "planning"
            if state.confirmation_status == "not_requested":
                state.confirmation_status = "pending"

    def decide(
        self,
        state: L2SessionState,
        *,
        stage_before: MIStage,
        analysis: SemanticAnalysis,
        response_override: str = "",
    ) -> StageDecision:
        pending = ScaleEngine.pending_item(state)
        if response_override:
            return StageDecision(
                stage_before=stage_before,
                stage_after=state.current_stage,
                transition_reason="输入需要澄清，阶段保持不变",
                response_goal=response_override,
                oars_techniques=["reflection", "open_question"],
                awaiting="scale_answer" if pending else "free_text",
                required_question=pending.prompt if pending else "",
            )
        if state.pending_scale_name:
            result = state.scale_results[state.pending_scale_name]
            definition = SCALE_CATALOG[state.pending_scale_name]
            if result.status == "pending_consent":
                return StageDecision(
                    stage_before=stage_before,
                    stage_after=state.current_stage,
                    transition_reason="发现与当前重点相关的量表，但尚未取得许可",
                    response_goal=(
                        f"解释{state.pending_scale_name}用于{definition.purpose}，"
                        "明确询问患者是否愿意进行"
                    ),
                    oars_techniques=["reflection", "open_question"],
                    awaiting="scale_consent",
                )
            if pending:
                anchors = "；".join(
                    f"{score}={label}" for score, label in pending.anchors.items()
                )
                question = f"{pending.prompt} 请按以下选项回答：{anchors}。"
                return StageDecision(
                    stage_before=stage_before,
                    stage_after=state.current_stage,
                    transition_reason="已获得量表许可，继续当前量表",
                    response_goal=(
                        "简短确认已收到上一题回答，不解释分数的临床含义，"
                        "不推断其他症状，再自然提出下一道量表问题"
                    ),
                    oars_techniques=["reflection", "open_question"],
                    awaiting="scale_answer",
                    required_question=question,
                )

        stage = state.current_stage
        if stage == "engaging":
            goal = "使用OARS理解患者最初的困扰、情绪和背景，邀请其继续叙述"
            awaiting = "free_text"
            techniques = ["complex_reflection", "open_question"]
        elif stage == "focusing":
            missing = missing_dimensions(state)
            if not state.focus_topics:
                goal = "总结患者已提到的困扰，邀请患者选择当前最想先谈的重点"
                awaiting = "focus_selection"
            elif len(state.symptoms) < 3:
                goal = "继续探索当前重点下的其他症状、频率和严重程度"
                awaiting = "free_text"
            elif diagnostic_missing := next(
                (
                    REQUIRED_DIAGNOSTIC_FIELD_BY_LABEL[label]
                    for label in missing
                    if label in REQUIRED_DIAGNOSTIC_FIELD_BY_LABEL
                ),
                None,
            ):
                goal = (
                    f"{diagnostic_missing.purpose}；一次只询问这一项，"
                    "不说明正在执行诊断排除"
                )
                awaiting = diagnostic_missing.field
            else:
                goal = "总结当前重点和已收集的信息，平滑邀请患者谈希望发生的变化"
                awaiting = "patient_goals"
            techniques = ["summary", "open_question"]
        elif stage == "evoking":
            if not state.patient_goals:
                goal = "探索患者最希望发生的变化及其个人意义"
                awaiting = "patient_goals"
            else:
                goal = "分别询问改变的重要性、信心和准备度，要求使用0到10分"
                awaiting = "motivation"
            techniques = ["complex_reflection", "open_question"]
        else:
            goal = "用已验证事实总结患者画像，请患者选择准确、修改或暂停"
            awaiting = "profile_confirmation"
            techniques = ["summary", "open_question"]

        changed = stage_before != state.current_stage
        return StageDecision(
            stage_before=stage_before,
            stage_after=state.current_stage,
            transition_reason=(
                "前一阶段的规则条件已满足，使用承接句平滑推进"
                if changed else "当前阶段仍有信息任务未完成"
            ),
            response_goal=goal,
            oars_techniques=techniques,
            awaiting=awaiting,
        )

    @staticmethod
    def _focusing_complete(state: L2SessionState) -> bool:
        if not state.focus_topics or len(state.symptoms) < 3:
            return False
        if any(
            label in REQUIRED_DIAGNOSTIC_FIELD_BY_LABEL
            for label in missing_dimensions(state)
        ):
            return False
        for result in state.scale_results.values():
            if result.status in {"pending_consent", "active"}:
                return False
        return True
