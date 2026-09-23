from layers.layer2.schemas import StageDecision


class MIResponseValidationError(ValueError):
    pass


class MIResponseValidator:
    SCALE_TERMS = ("量表", "问卷", "PHQ-9", "GAD-7", "PCL-5", "Y-BOCS", "ISI", "MDQ")
    FORBIDDEN = (
        "你被确诊",
        "确诊为",
        "诊断为",
        "DSM-5",
        "处方",
        "药物剂量",
        "参与阶段",
        "聚焦阶段",
        "唤出阶段",
        "计划阶段",
        "进入下一阶段",
        "engaging",
        "focusing",
        "evoking",
        "planning",
    )

    def assert_valid(
        self,
        reply: str,
        decision: StageDecision,
        *,
        active_safety_controls: list[str] | None = None,
        patient_message: str = "",
        scale_discussion_started: bool = False,
    ) -> None:
        if not reply.strip():
            raise MIResponseValidationError("MI回复为空")
        if len(reply) > 1200:
            raise MIResponseValidationError("MI回复过长")
        if decision.required_question and decision.required_question not in reply:
            raise MIResponseValidationError("量表问题或选项被模型改写")
        found = [term for term in self.FORBIDDEN if term.lower() in reply.lower()]
        if found:
            raise MIResponseValidationError("L2回复越界：" + ",".join(found))
        if (
            not scale_discussion_started
            and decision.awaiting not in {"scale_consent", "scale_answer"}
            and not decision.required_question
        ):
            answer = reply.casefold()
            patient = patient_message.casefold()
            if any(
                term.casefold() in answer and term.casefold() not in patient
                for term in self.SCALE_TERMS
            ):
                raise MIResponseValidationError("临床访谈尚未完成，不得主动提出量表")
        controls = set(active_safety_controls or [])
        if "avoid_trauma_probing" in controls:
            trauma_probes = (
                "详细描述创伤",
                "详细讲述那件事",
                "回忆事件细节",
                "发生了什么细节",
                "重新经历那件事",
            )
            if any(term in reply for term in trauma_probes):
                raise MIResponseValidationError("回复违反避免创伤深挖的安全控制")
