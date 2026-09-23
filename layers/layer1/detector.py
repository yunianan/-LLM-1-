from .keyword_safety import keyword_detect
from .risk_classifier import llm_classify
from .jailbreak_safety import jailbreak_detect
from .risk_fusion import fuse_results
from .policy_executor import execute_safety_policy
from .guard import L1ExitGuard


def safety_detect(text):
    keyword_result = keyword_detect(text)
    llm_result = llm_classify(text)
    if llm_result.get("degraded") is True:
        return {
            "layer_id": "L1",
            "status": "error",
            "error_code": "L1_LLM_UNAVAILABLE",
            "message": "安全评估模型不可用",
            "terminal": True,
            "can_call_l2": False,
            "details": {
                "attempts": llm_result.get("attempts", 0),
                "failure_reason": llm_result.get("failure_reason", ""),
                "rule_score": keyword_result.get("rule_score", 0),
                "rule_risk_types": keyword_result.get("risk_types", []),
            },
            "evidence_refs": (
                list(keyword_result.get("evidence_refs") or [])
                + list(llm_result.get("evidence_refs") or [])
            ),
        }
    jailbreak_result = jailbreak_detect(text)

    final_result = fuse_results(
        keyword_result,
        llm_result,
        jailbreak_result
    )

    final_result["execution"] = execute_safety_policy(final_result)
    final_result["guard"] = L1ExitGuard().assert_valid(final_result)

    return final_result
