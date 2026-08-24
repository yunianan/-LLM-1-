"""接口契约验证：L1 SafetyPassport / L3 DiagnosticReport / L5 审核输出 / L4 端到端。

用法：python tools/verify_interfaces.py
LLM 调用全部 mock，无网络。
"""

import json
import sys

import shared.llm_client as llm_client


def fake_ask_llm(prompt, model="deepseek-chat", temperature=0):
    if "输出安全审核系统" in prompt:
        if "你得了" in prompt or "建议你去开药" in prompt:
            return json.dumps({
                "harm": "fail", "boundary": "fail", "quality": "pass",
                "fail_reason": "回复含确定性诊断措辞或药物建议",
                "fix_instruction": "删除确诊式表述与药物建议，使用'初步印象'措辞。",
            })
        return json.dumps({
            "harm": "pass", "boundary": "pass", "quality": "pass",
            "fail_reason": "", "fix_instruction": "",
        })
    if "判断风险等级" in prompt:
        return json.dumps({
            "risk_level": 1, "score": 1, "risk_types": [],
            "confidence": 0.9, "reason": "低风险，无紧急信号",
        })
    return "{}"


llm_client.ask_llm = fake_ask_llm

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def require_keys(name, data, keys):
    missing = [k for k in keys if k not in data]
    check(f"{name} 必含字段 {keys}", not missing, f"缺失: {missing}")


print("=" * 60)
print("L1: SafetyPassport（接口一）")
print("=" * 60)

from layers.layer1.detector import safety_detect

SP_KEYS = [
    "layer_id", "timestamp", "safety_flag", "risk_level", "risk_types",
    "crisis_flag", "sensitive_topics", "summary", "structured_labels",
    "evidence_refs", "confidence", "flags", "instruction_to_next",
]

sp = safety_detect("最近工作压力很大，晚上总是睡不好。")
require_keys("L1 输出", sp, SP_KEYS)
check("L1 layer_id == L1", sp["layer_id"] == "L1")
check("L1 safety_flag 枚举 pass/warn/block", sp["safety_flag"] in ("pass", "warn", "block"))
check("L1 evidence_refs 每项含 source/finding/basis",
      all({"source", "finding", "basis"} <= set(e) for e in sp["evidence_refs"]))
check("L1 summary 为自然语言", isinstance(sp["summary"], str) and len(sp["summary"]) > 5)

sp_crisis = safety_detect("我不想活了，我想结束生命。")
check("L1 危机级 risk_level>=3", sp_crisis["risk_level"] >= 3, f"got {sp_crisis['risk_level']}")
check("V1-2: risk_level=3 → crisis_flag=true",
      sp_crisis["risk_level"] >= 3 and sp_crisis["crisis_flag"] is True)
check("V1-1 枚举: crisis → safety_flag=block",
      sp_crisis["safety_flag"] == "block", f"got {sp_crisis['safety_flag']}")

sp_trauma = safety_detect("小时候家里经常吵架，爸妈关系很差。")
check("L1 sensitive_topics 含 trauma", "trauma" in sp_trauma["sensitive_topics"],
      f"got {sp_trauma['sensitive_topics']}")
check("L1 创伤标记 → avoid_trauma_probing",
      "avoid_trauma_probing" in sp_trauma["flags"])

from l4.schemas.instructions import SafetyPassport as L4SafetyPassport

L4SafetyPassport(
    risk_level=sp["risk_level"],
    sensitive_topics=sp["sensitive_topics"],
    flags=sp["flags"],
)
check("L1 输出可被 L4 SafetyPassport 消费", True)

print("=" * 60)
print("L3: DiagnosticReport（接口三）")
print("=" * 60)

from layers.layer3.api import diagnose

profile = {
    "PHQ9": 16, "GAD7": 9, "MDQ": 3, "PCL5": 18, "ISI": 14,
    "Y_BOCS": 0, "ADNM": 0,
}
safety_ctx = {
    "risk_level": 1,
    "sensitive_topics": ["trauma"],
    "l1_flags": ["monitor_suicidal_ideation", "avoid_trauma_probing"],
}

DR_KEYS = [
    "layer_id", "timestamp", "diagnosis", "verifier_score", "verifier_routing",
    "recommended_therapy", "severity_level", "clinical_notes_for_L4", "summary",
    "structured_labels", "evidence_refs", "confidence", "flags", "instruction_to_next",
]
report = diagnose(profile, safety_ctx)
require_keys("L3 输出", report, DR_KEYS)
check("L3 layer_id == L3", report["layer_id"] == "L3")

primary = report["diagnosis"]["primary"]
check("L3 primary 含 disorder/confidence/severity/dsm5_criteria_met",
      {"disorder", "confidence", "severity", "dsm5_criteria_met"} <= set(primary))
check("L3 verifier_routing 枚举", report["verifier_routing"] in ("pass", "conditional", "reject"))
check("L3 recommended_therapy 含 primary/addon/contraindications",
      {"primary", "addon", "contraindications"} <= set(report["recommended_therapy"]))
check("L3 structured_labels 含 therapy_routing/wording_constraint/risk_monitoring",
      {"therapy_routing", "wording_constraint", "risk_monitoring"} <= set(report["structured_labels"]))
check("L3 wording_constraint 为'初步印象'", "初步印象" in report["structured_labels"]["wording_constraint"])

met = primary["dsm5_criteria_met"]
check("V3-1: primary.confidence >= 0.5", primary["confidence"] >= 0.5, f"got {primary['confidence']}")
check("V3-2: MDD dsm5_criteria_met >= 5", len(met) >= 5, f"got {len(met)}: {met}")
secondary = {s["disorder"] for s in report["diagnosis"]["secondary"]}
ruled_out = {r["disorder"] for r in report["diagnosis"]["ruled_out"]}
check("V3-3: ruled_out 不在 primary/secondary",
      ruled_out.isdisjoint({primary["disorder"]} | secondary))

from l4.schemas.l3_report import DiagnosticReport

parsed = DiagnosticReport.model_validate(report)
check("L3 输出可被 L4 DiagnosticReport schema 解析", parsed.diagnosis.primary.disorder == "MDD")
check("L4 anchor: therapy_type 取自 recommended_therapy.primary",
      parsed.to_anchor().therapy_type == report["recommended_therapy"]["primary"])

print("=" * 60)
print("L5: ContextualSafetyContext 输入 → 文档输出格式（接口四）")
print("=" * 60)

from layers.layer5.reviewer import review_response

ctx = {
    "l4_raw_output": "根据我们目前的评估，初步印象是中度抑郁，你觉得这个判断符合你的感受吗？",
    "l4_substage": "L4-L2",
    "therapy_type": "CBT",
    "diagnosis_context": {
        "primary": "MDD (moderate, confidence=0.75)",
        "secondary": [],
        "suicidal_ideation": "passive_only",
    },
    "risk_context": {"risk_level": 1, "sensitive_topics": ["trauma"]},
}

L5_KEYS = ["layer_id", "timestamp", "harm", "boundary", "quality",
           "fail_reason", "fix_instruction", "evidence_refs", "confidence",
           "flags", "instruction_to_next"]
out = review_response(ctx)
require_keys("L5 输出", out, L5_KEYS)
check("L5 layer_id == L5", out["layer_id"] == "L5")
check("L5 harm/boundary/quality 枚举", {out["harm"], out["boundary"], out["quality"]} <= {"pass", "fail"})
check("L5 fail_reason/fix_instruction 为字符串",
      isinstance(out["fail_reason"], str) and isinstance(out["fix_instruction"], str))

ctx_bad = dict(ctx)
ctx_bad["l4_raw_output"] = "你得了中度抑郁，建议你去开药。"
out_bad = review_response(ctx_bad)
check("L5 越界输出被拦截 (boundary=fail)", out_bad["boundary"] == "fail", f"got {out_bad['boundary']}")
check("L5 fail 时 flags 标记", "manual_review_required" in out_bad["flags"], f"got {out_bad['flags']}")

from l4.schemas.review import ReviewResult

ReviewResult(
    harm=out["harm"], boundary=out["boundary"], quality=out["quality"],
    fail_reason=out["fail_reason"], fix_instruction=out["fix_instruction"],
)
check("L5 输出可被 L4 ReviewResult schema 解析", True)

print("=" * 60)
print("端到端：L1 → L3 → L4（含 L5 审核回环）")
print("=" * 60)

from l4.config import Settings
from l4.llm.mock import MockBackend
from l4.review import L5Reviewer
from l4.service import L4Service, SessionRequest, TurnRequest
from l4.schemas.instructions import (
    BoundaryInstruction, RoleInstruction, SafetyPassport, TurnInstruction,
)

service = L4Service(
    Settings(_env_file=None),
    backend=MockBackend(),
    reviewer=L5Reviewer(review_response),
)
session_id = service.create_session(SessionRequest(
    role=RoleInstruction(identity="你是一名受过CBT训练的心理治疗师", style="温暖、共情、结构化"),
    report=parsed,
    boundary=BoundaryInstruction(
        absolute_bans=["禁止开药", "禁止诊断新疾病", "禁止预测未来", "替代医生", "谈论其他患者"],
        safety_trigger="用户表达自杀意图→立即终止对话，触发危机协议",
    ),
    safety=SafetyPassport(risk_level=1, sensitive_topics=["trauma"],
                          flags=["monitor_suicidal_ideation"]),
))
outcome = service.handle_turn(TurnRequest(
    session_id=session_id,
    user_message="我最近工作压力很大，晚上总是睡不好。",
    turn=TurnInstruction(goal="建立共情与治疗联盟", technique="开放式提问+反映",
                         forbidden=["建议", "确诊"], force_substage="L4-L1"),
))
check("L4 回复非空", bool(outcome.reply))
check("L4 l4_substage 枚举", outcome.l4_substage in ("L4-L1", "L4-L2", "L4-L3", "L4-L4"))
check("L4 审核通过（真实 L5 逻辑 + mock LLM）", outcome.fallback_used is False,
      f"review={[r.model_dump() for r in outcome.review_history]}")
check("L4 summary 含 evidence_refs", len(outcome.summary.evidence_refs) > 0)
check("L5 上下文含 trauma 敏感标记",
      "trauma" in outcome.l5_context.risk_context.sensitive_topics)

print()
print(f"结果: {PASS} 通过 / {FAIL} 失败")
sys.exit(1 if FAIL else 0)
