from layers.layer2.schemas import DialogueResult, ExpectedDialogueInput
from layers.layer2.service import L2AssessmentService


class L2DialogueAdapter:
    """把患者自然语言交给 L2；不在适配器内复制任何解析规则。"""

    def __init__(self, service: L2AssessmentService) -> None:
        self.service = service

    def handle_message(
        self,
        session_id: str,
        patient_text: str,
        *,
        expected_input: ExpectedDialogueInput = "auto",
        message_id: str | None = None,
    ) -> DialogueResult:
        return self.service.handle_message(
            session_id,
            patient_text,
            expected_input=expected_input,
            message_id=message_id,
        )
