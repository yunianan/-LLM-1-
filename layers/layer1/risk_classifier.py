import json

from shared.llm_client import ask_llm
from shared.prompts.risk_prompt import build_prompt


def llm_classify(text):
    prompt = build_prompt(text)
    response = ask_llm(prompt)
    data = json.loads(response)

    return {
        "llm_score": data["score"],
        "risk_types": data.get("risk_types", []),
        "confidence": data.get("confidence", 0.8),
        "evidence_refs": [
            {
                "source": "DeepSeek_API",
                "finding": f"risk_level={data['risk_level']}",
                "basis": data["reason"]
            }
        ]
    }
