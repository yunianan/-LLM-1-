from layers.layer3.instructions.turn_planner import TurnInstructionPlanner, TurnProposal


class ControlledTreatmentModel:
    def __init__(self, **overrides):
        self.overrides = overrides
        self.calls = []

    def propose(self, context):
        self.calls.append(context)
        source = context["current_source_id"]
        message = context["sources"][source]
        records = context["progress"]
        last = records[-1] if records else None
        values = dict(
            goal=f"探索患者本轮提到的困扰：{message}",
            technique_code="reflection",
            stage=last["instruction"]["force_substage"] if last else "L4-L1",
            link_previous=f"衔接上轮：{last['user_message']}" if last else "",
            rationale="依据当前患者原话确定本轮目标",
            evidence=[{"source_id": source, "quote": message}],
        )
        values.update(self.overrides)
        return TurnProposal.model_validate(values)


def controlled_turn(service, session_id, user_message, **overrides):
    sd = service.store.get(session_id)
    return TurnInstructionPlanner(ControlledTreatmentModel(**overrides)).plan(
        session_id=session_id, instructions=sd.instructions(),
        records=sd.treatment_records(), user_message=user_message, safety={},
    )
