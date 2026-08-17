from l4.llm.base import LLMBackend, Message


class LocalModelBackend(LLMBackend):
    """本地训练模型适配器占位。

    TODO: 本地模型训练完成后，在此实现 vLLM / transformers 接入
    （如 OpenAI 兼容的 vLLM server 或 transformers pipeline），
    并保持 chat() 签名不变。L4 其他代码无需任何改动。
    """

    def __init__(self, model_path: str = "") -> None:
        raise NotImplementedError(
            "LocalModelBackend 尚未实现：本地模型训练完成后接入 vLLM/transformers。"
        )

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError
