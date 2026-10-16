"""LLM 适配器工厂。根据 ``Settings`` 决定用离线伪模型还是真实模型。"""

from __future__ import annotations

from patient_agent.llm.base import LLMClient
from patient_agent.llm.openai_compat import OpenAICompatLLM
from patient_agent.llm.rule_based import RuleBasedLLM

__all__ = ["LLMClient", "OpenAICompatLLM", "RuleBasedLLM", "build_llm_client"]


# ─────────────────────────────────────────────────────────────────────────────
# 【接入点 7】LLM 选择开关（离线伪模型 ←→ 真实模型）
# 接入什么：运行期决定用 RuleBasedLLM 还是 OpenAICompatLLM。
# 怎么打开：config.Settings.llm_mode（环境变量 PATIENT_AGENT_LLM_MODE=offline|auto|api）。
#           offline 恒用伪模型；auto 有 key 才用真模型；api 强制真模型、缺 key 直接报错。
# 新增一家模型（非 OpenAI 兼容协议）：在 llm/ 下加一个适配器，在这里加一个分支。
# ─────────────────────────────────────────────────────────────────────────────


def build_llm_client(settings: object) -> LLMClient:
    """按配置造客户端。

    ``settings`` 只需有 ``use_real_llm`` 与 ``openai_*`` 字段（即 ``config.Settings``），
    这里用 duck typing 以免 llm 包反向依赖 config，方便单测塞桩。
    """

    use_real = getattr(settings, "use_real_llm", False)
    if use_real:
        client = OpenAICompatLLM(
            base_url=getattr(settings, "openai_base_url"),
            api_key=getattr(settings, "openai_api_key"),
            model=getattr(settings, "openai_model"),
            timeout=getattr(settings, "openai_timeout", 60.0),
            temperature=getattr(settings, "openai_temperature", 0.7),
        )
        if client.is_available():
            return client
    return RuleBasedLLM()
