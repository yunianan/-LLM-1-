from .keyword_safety import keyword_detect
from .risk_classifier import llm_classify
from .jailbreak_safety import jailbreak_detect
from .risk_fusion import fuse_results


def safety_detect(text):
    keyword_result = keyword_detect(text)
    llm_result = llm_classify(text)
    jailbreak_result = jailbreak_detect(text)

    final_result = fuse_results(
        keyword_result,
        llm_result,
        jailbreak_result
    )

    return final_result
