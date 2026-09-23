from __future__ import annotations

from dataclasses import dataclass

from .models import ASQState


ASQ_QUESTIONS = {
    1: "在过去几周，您是否希望自己已经死去？请回答“是”或“否”。",
    2: "在过去几周，您是否觉得如果自己死了，对您自己或家人都会更好？请回答“是”或“否”。",
    3: "在过去一周，您是否产生过杀死自己的想法？请回答“是”或“否”。",
    4: "您是否曾经尝试过自杀？请回答“是”或“否”。",
    5: "您现在是否正在产生杀死自己的想法？请回答“是”或“否”。",
}

INTRO = (
    "在这次心理评估中，我们始终重视您的安全。"
    "疾病、生活压力和其他困扰有时可能引发强烈的情绪，甚至让人产生自杀相关的想法。"
    "为了进一步确认您目前是否安全，接下来我会进行一项简短的自杀风险筛查问卷（ASQ）。"
    "请根据真实情况回答“是”或“否”。\n"
)


@dataclass(frozen=True)
class ASQTurnResult:
    reply: str
    status: str


def start_asq(
    state: ASQState,
    *,
    trigger: str,
    supportive_opening: str = "",
) -> str:
    state.status = "in_progress"
    state.trigger = "signal" if trigger == "signal" else "baseline"
    state.current_question = 1
    state.awaiting_attempt_timing = False
    state.answers = {}
    state.most_recent_attempt_timing = ""
    opening = supportive_opening.strip()
    prefix = f"{opening}\n" if opening else ""
    return prefix + INTRO + ASQ_QUESTIONS[1]


def _answer(text: str) -> bool | None | str:
    value = text.strip().lower()
    if any(token in value for token in ("拒绝回答", "不想回答", "不方便回答")):
        return "refused"
    yes = {"是", "有", "是的", "有过", "想过", "yes", "y"}
    no = {"否", "没有", "不是", "从来没有", "没想过", "no", "n"}
    if value in yes:
        return True
    if value in no:
        return False
    return None


def handle_asq_answer(state: ASQState, patient_text: str) -> ASQTurnResult:
    if state.status != "in_progress":
        raise ValueError("ASQ当前不在进行中")

    if state.awaiting_attempt_timing:
        timing = patient_text.strip()
        if not timing:
            return ASQTurnResult("请告诉我最近一次尝试大约发生在什么时候。", state.status)
        state.most_recent_attempt_timing = timing
        state.awaiting_attempt_timing = False
        state.current_question = 5
        return ASQTurnResult(ASQ_QUESTIONS[5], state.status)

    parsed = _answer(patient_text)
    if parsed is None:
        return ASQTurnResult(
            "为了按照ASQ标准准确记录，请明确回答“是”或“否”。\n"
            + ASQ_QUESTIONS[state.current_question],
            state.status,
        )
    if parsed == "refused":
        state.status = "refused"
        return ASQTurnResult("已记录您暂时不回答安全筛查问题。", state.status)

    question = state.current_question
    state.answers[question] = bool(parsed)
    if question < 4:
        state.current_question += 1
        return ASQTurnResult(ASQ_QUESTIONS[state.current_question], state.status)
    if question == 4:
        if parsed:
            state.awaiting_attempt_timing = True
            return ASQTurnResult("最近一次尝试大约发生在什么时候？", state.status)
        if any(state.answers.get(index) is True for index in range(1, 5)):
            state.current_question = 5
            return ASQTurnResult(ASQ_QUESTIONS[5], state.status)
        state.status = "negative"
        return ASQTurnResult("谢谢您完成安全筛查。我们可以继续了解您当前的困扰。", state.status)

    state.status = "acute_positive" if parsed else "non_acute_positive"
    return ASQTurnResult("感谢您如实告诉我。接下来我需要进一步确认您的安全状况。", state.status)
