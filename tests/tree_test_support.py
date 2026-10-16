"""Controlled node responses for engineering tests; not clinical gold cases."""
import json


def node_payload(prompt, labels=("D013",), ref="ev-mood", overrides=None):
    batch = json.JSONDecoder().raw_decode(prompt.split("当前节点 JSON：", 1)[1].lstrip())[0]
    answers = []
    for node in batch:
        selected = bool(set(node["disorders"]) & set(labels))
        status = "pending"
        if node["stage"] == "screening":
            status = "met" if selected else "not_met"
        if node["stage"] == "anchor":
            status = "met" if selected else "not_met"
        status = (overrides or {}).get(node["node_id"], status)
        answers.append({"node_id": node["node_id"], "status": status,
                        "rationale": "controlled engineering fixture",
                        "evidence_refs": [ref] if status != "pending" else [],
                        "symptom_ids": node.get("symptom_pool", [])[:1] if status == "met" else [],
                        "confidence": 0.75})
    return {"nodes": answers}
