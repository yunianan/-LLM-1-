"""逐层调用真实 LLM 的独立手工测试入口。1


在 VS Code 中右击本文件，选择“在终端中运行 Python 文件”；也可以运行：
    python live_layer_test.py l3

本文件只为目标层提供固定的虚构上游输入，不串行执行前置层。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2)


def _configure_real_api() -> None:
    """读取本机配置；没有配置时只写入当前测试进程的环境变量。"""
    from dotenv import load_dotenv
    from shared.llm_client import get_api_key, get_base_url, get_model

    load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
    if not get_api_key():
        platform = input("选择 API 平台：1=DeepSeek 官方，2=阿里云百炼 [1]：").strip() or "1"
        if platform == "2":
            base_url = input(
                "请输入阿里 OpenAI 兼容 base_url（不要包含 chat/completions）："
            ).strip()
            model = input("请输入阿里控制台中的模型 ID：").strip()
            parsed = urlsplit(base_url)
            if (
                not base_url
                or not model
                or parsed.scheme != "https"
                or not parsed.hostname
                or parsed.path.rstrip("/").endswith("/chat/completions")
            ):
                raise ValueError("阿里平台的 base_url 或模型 ID 不合法")
            os.environ["LLM_BASE_URL"] = base_url.rstrip("/")
            os.environ["LLM_MODEL"] = model
        elif platform == "1":
            os.environ.setdefault("LLM_BASE_URL", "https://api.deepseek.com")
            os.environ.setdefault("LLM_MODEL", "deepseek-chat")
        else:
            raise ValueError("平台只能选择 1 或 2")

        key = input("请输入 API Key（明文显示，仅保存在本次进程中）：").strip()
        if not key:
            raise ValueError("没有提供 API Key")
        os.environ["LLM_API_KEY"] = key

    endpoint = urlsplit(get_base_url()).hostname or "无法识别"
    print(f"\n[连接] endpoint={endpoint}；model={get_model()}；使用真实 API")


def _run_timed(label: str, operation: Callable[[], Any]) -> Any:
    print(f"\n[{label}] 正在调用真实 LLM……", flush=True)
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[{label}] 失败，耗时 {elapsed:.2f}s")
        _show_safe_exception_chain(exc)
        try:
            setattr(exc, "_live_error_shown", True)
        except Exception:
            pass
        raise
    elapsed = time.perf_counter() - started
    print(f"[{label}] 完成，耗时 {elapsed:.2f}s")
    return result


def _show_safe_exception_chain(error: BaseException) -> None:
    """显示错误类型和安全元数据，不打印请求体、患者原话或密钥。"""
    current: BaseException | None = error
    seen: set[int] = set()
    level = 0
    safe_messages = {
        "CLINICRModelUnavailable",
        "L2ModelUnavailable",
        "L2ModelOutputInvalid",
        "L3ToL4ValidationError",
    }
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        info: dict[str, Any] = {
            "level": level,
            "type": type(current).__name__,
        }
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            info["http_status"] = status
        if type(current).__name__ in safe_messages:
            info["message"] = str(current)
        if type(current).__name__ == "ValidationError" and hasattr(current, "errors"):
            info["validation_errors"] = [
                {"path": list(item["loc"]), "type": item["type"]}
                for item in current.errors(
                    include_input=False,
                    include_context=False,
                    include_url=False,
                )[:20]
            ]
        print("[安全错误链] " + json.dumps(info, ensure_ascii=False))
        current = current.__cause__ or current.__context__
        level += 1


def test_l1() -> None:
    """持续运行 L1，直到允许进入 L2 或给出安全交接结果。"""
    from layers.layer1.detector import safety_detect, start_baseline_asq

    session_id = "live-l1-" + uuid.uuid4().hex
    text = input(
        "患者消息 [最近工作压力很大，晚上睡不好，但我没有伤害自己的想法。]"
        "（/baseline 测常规 ASQ，/exit 退出）："
    ).strip() or "最近工作压力很大，晚上睡不好，但我没有伤害自己的想法。"
    turn = 0
    while text != "/exit":
        turn += 1
        result = _run_timed(
            f"L1 第{turn}轮",
            lambda: (
                start_baseline_asq(session_id=session_id, message_id=uuid.uuid4().hex)
                if text == "/baseline"
                else safety_detect(
                    text, session_id=session_id, message_id=uuid.uuid4().hex
                )
            ),
        )
        if result is None:
            raise RuntimeError("当前会话无法启动常规 ASQ")
        if result.get("status") == "error":
            print("\n[L1 错误结果]\n" + _json(result))
            raise RuntimeError("L1 未完成，模型或 API 返回错误")
        execution = result.get("execution") or {}
        print("\n系统：" + str(execution.get("user_response") or ""))
        print(
            "[开发调试] 路由：{0}；等待：{1}".format(
                execution.get("route"), execution.get("waiting_for") or "无"
            )
        )
        if execution.get("route") == "l2" or execution.get("resume_l2_next_turn"):
            print("\n[L1 出口：允许下一条患者消息进入 L2]\n" + _json(result))
            return
        if execution.get("route") == "safety":
            print("\n[L1 出口：普通疗程暂停，已输出安全交接指引]\n" + _json(result))
            return
        if execution.get("route") not in {"l1", "security"}:
            raise RuntimeError("L1 返回了未知路由，未到达本层出口")
        text = input("你（/exit 可中止测试）：").strip()
        while not text:
            text = input("请输入回答（/exit 可中止测试）：").strip()
    print("L1 测试由用户中止，未到达本层出口。")


def test_l2() -> None:
    """使用真实 L2 模型持续评估，直到画像确认且通过 L2→L3 守卫。"""
    from layers.layer1.service import L1SafetyService
    from layers.layer2 import L2AssessmentService, L2ToL3Adapter
    from layers.layer2.llm import APIL2Model
    from orchestration.schemas import PipelineSession
    from patient_presentation.developer_profile import DeveloperProfileRenderer

    session_id = "live-l2-" + uuid.uuid4().hex
    # 独立测试只模拟上游已放行，不调用 L1 模型；真实患者请走 main.py。
    passport = L1SafetyService().handle_message(
        "虚构测试会话的固定上游输入",
        session_id=session_id,
        message_id=uuid.uuid4().hex,
        classifier=lambda *_args: {
            "analysis": {
                "suicide_signal_detected": False,
                "jailbreak_detected": False,
                "safe_to_enter_l2": True,
                "confidence": 1.0,
                "evidence_quotes": [],
                "jailbreak_evidence": [],
                "asq_transition_support": "",
                "summary": "独立 L2 测试的固定上游安全结果",
            }
        },
    )
    service = L2AssessmentService(model=APIL2Model(), require_fresh_passport=False)
    service.create_session(passport, session_id=session_id)
    adapter = L2ToL3Adapter()
    profile_renderer = DeveloperProfileRenderer(
        PROJECT_ROOT / "artifacts" / "developer_profiles"
    )
    profile_session = PipelineSession(
        session_id=session_id, l2_session_id=session_id, l1_passport=passport,
        clinical_intake=service.build_clinical_intake(session_id),
    )
    profile_path = profile_renderer.render(profile_session)
    print("本项仅使用固定 L1 测试凭证；L2 的语义理解与 MI 回复调用真实 API。")
    print("持续对话直到患者确认画像并通过 L2→L3 守卫；/exit 可中止测试。")
    print(f"[开发者画像，仅限本机调试] {profile_path}")
    print("/profile 用浏览器打开画像；每轮更新后刷新页面。HTML 仅供查看，不能恢复会话。")
    text = input(
        "患者消息 [最近一个多月我总觉得没劲，工作也受到影响，想先改善睡眠。]："
    ).strip() or "最近一个多月我总觉得没劲，工作也受到影响，想先改善睡眠。"
    expected_input = "auto"
    turn = 0
    while text != "/exit":
        if text == "/profile":
            try:
                opened = webbrowser.open(profile_path.as_uri())
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                print(f"请用浏览器手动打开：{profile_path}")
            text = input("你（/profile 查看画像；/exit 中止测试）：").strip()
            while not text:
                text = input("请输入回答（/profile 查看画像；/exit 中止测试）：").strip()
            continue
        turn += 1
        result = _run_timed(
            f"L2 第{turn}轮（语义提取＋规则处理＋MI 回复）",
            lambda: service.handle_message(
                session_id, text, expected_input=expected_input
            ),
        )
        print("\n系统：" + result.reply)
        if result.error_code in {"L2_MODEL_UNAVAILABLE", "L2_MODEL_OUTPUT_INVALID"}:
            raise RuntimeError(f"L2 未完成：{result.error_code}")
        if not result.accepted:
            print(f"[开发调试] 当前消息未被接受：{result.error_code or 'unknown'}")
            expected_input = result.awaiting or expected_input
        else:
            intake = service.build_clinical_intake(session_id)
            profile_session.clinical_intake = intake
            profile_session.expected_input = result.awaiting or "auto"
            profile_session.route = (
                "PAUSED_BY_PATIENT" if result.confirmation_decision == "pause"
                else "AWAITING_PATIENT_CONFIRMATION"
                if intake.patient_confirmation == "pending"
                and intake.information_sufficiency == "sufficient"
                else "L2_INTERVIEW"
            )
            profile_session.touch()
            profile_path = profile_renderer.render(profile_session)
            print(f"[开发者画像已更新] {profile_path}")
            if result.confirmation_decision == "pause":
                print("L2 已按患者选择暂停，未交接至 L3。")
                return
            if result.confirmation_decision == "revise":
                print("[开发调试] 请说明画像中需要修改的具体信息。")
                expected_input = "free_text"
            elif intake.information_sufficiency == "sufficient":
                if intake.patient_confirmation == "confirmed":
                    receipt = adapter.guard.verify(intake)
                    if receipt["status"] == "pass":
                        print("\n[L2 出口：画像已确认，L2→L3 守卫通过]")
                        print("\n[守卫回执]\n" + _json(receipt))
                        print("\n[L2 患者画像]\n" + _json(adapter.to_profile(intake)))
                        return
                    print("[开发调试] L2→L3 守卫未通过：" + "；".join(receipt["violations"]))
                    service.confirm_profile(session_id, "revise")
                    expected_input = "free_text"
                else:
                    print("\n[待患者确认的画像]\n" + _json(intake))
                    expected_input = "profile_confirmation"
            else:
                print("[开发调试] 待补充：" + "；".join(intake.missing_dimensions))
                expected_input = result.awaiting or "auto"
            if result.l2_turn and result.l2_turn.pending_scale_item:
                pending = result.l2_turn.pending_scale_item
                print(f"[开发调试] 当前量表题：{pending.scale_name} 第{pending.item_number}题")
                expected_input = "scale_answer"
        text = input("你（/exit 可中止测试）：").strip()
        while not text:
            text = input("请输入回答（/exit 可中止测试）：").strip()
    print("L2 测试由用户中止，未到达 L2→L3 出口。")


def _fictional_l3_profile() -> dict[str, Any]:
    """虚构且证据标识完整的 L2→L3 输入，不经过 L1/L2。"""
    return {
        "PHQ9": 17,
        "PHQ9_items": {
            "item1": 3,
            "item2": 3,
            "item3": 2,
            "item4": 3,
            "item5": 2,
            "item6": 2,
            "item7": 1,
            "item8": 1,
            "item9": 0,
        },
        "GAD7": 6,
        "MDQ": 2,
        "PCL5": 8,
        "ISI": 12,
        "Y_BOCS": 2,
        "symptom_duration_days": 45,
        "functional_impairment": True,
        "mania_or_hypomania_history": False,
        "substance_or_medical_cause": False,
        "psychotic_symptoms": False,
        "differential_diagnoses_assessed": [
            "MDD",
            "GAD",
            "PTSD",
            "OCD",
            "Insomnia",
            "Panic",
            "SocialAnxiety",
            "Adjustment",
            "SomaticSymptom",
            "Bipolar",
        ],
        "evidence_conflicts": [],
        "key_symptoms_extracted": [
            {
                "symptom": "depressed_mood",
                "sources": ["PHQ9_item2=3", "ev-mood"],
                "patient_quotes": ["这一个多月几乎每天都很低落"],
                "attributes": {"duration_days": 45},
            },
            {
                "symptom": "anhedonia",
                "sources": ["PHQ9_item1=3", "ev-interest"],
                "patient_quotes": ["以前喜欢的篮球现在完全不想碰"],
                "attributes": {},
            },
            {
                "symptom": "fatigue",
                "sources": ["PHQ9_item4=3", "ev-fatigue"],
                "patient_quotes": ["每天都觉得没力气"],
                "attributes": {},
            },
            {
                "symptom": "insomnia",
                "sources": ["PHQ9_item3=2", "ISI=12", "ev-sleep"],
                "patient_quotes": ["晚上经常要一个多小时才能睡着"],
                "attributes": {},
            },
            {
                "symptom": "worthlessness",
                "sources": ["PHQ9_item6=2", "ev-worth"],
                "patient_quotes": ["我总觉得自己什么都做不好"],
                "attributes": {},
            },
        ],
        "emotions": [
            {
                "value": "低落和无助",
                "sources": ["ev-mood"],
                "patient_quotes": ["这一个多月几乎每天都很低落"],
            }
        ],
        "cognitive_patterns": [
            {
                "value": "我什么都做不好",
                "sources": ["ev-worth"],
                "patient_quotes": ["我总觉得自己什么都做不好"],
            }
        ],
        "key_narratives": [
            {
                "value": "症状持续约45天",
                "sources": ["ev-duration"],
                "patient_quotes": ["这一个多月"],
            }
        ],
        "functional_impairments": [
            {
                "value": "工作效率显著下降",
                "sources": ["ev-function"],
                "patient_quotes": ["工作任务经常拖到最后也做不完"],
            }
        ],
        "patient_goals": ["恢复精力并改善工作状态"],
        "motivation": {
            "importance": 8,
            "confidence": 6,
            "readiness": 7,
            "sources": ["ev-goal"],
        },
        "evidence_nodes": [
            {
                "evidence_id": evidence_id,
                "source_type": "patient_message",
                "message_id": "fictional-message-1",
                "turn_index": 1,
                "quote": quote,
                "target_path": target,
                "created_by": "llm_candidate_rule_verified",
                "parent_ids": [],
            }
            for evidence_id, quote, target in (
                ("ev-mood", "这一个多月几乎每天都很低落", "symptoms.depressed_mood"),
                ("ev-interest", "以前喜欢的篮球现在完全不想碰", "symptoms.anhedonia"),
                ("ev-fatigue", "每天都觉得没力气", "symptoms.fatigue"),
                ("ev-sleep", "晚上经常要一个多小时才能睡着", "symptoms.insomnia"),
                ("ev-worth", "我总觉得自己什么都做不好", "symptoms.worthlessness"),
                ("ev-duration", "这一个多月", "diagnostic_context.symptom_duration_days"),
                ("ev-function", "工作任务经常拖到最后也做不完", "diagnostic_context.functional_impairment"),
                ("ev-mania", "我从来没有连续几天异常兴奋或少睡也不累", "diagnostic_context.mania_or_hypomania_history"),
                ("ev-substance", "这些变化不是在饮酒或用药后出现的", "diagnostic_context.substance_or_medical_cause"),
                ("ev-psychotic", "没有听到别人听不到的声音，也没有感觉被监视", "diagnostic_context.psychotic_symptoms"),
                ("ev-goal", "我希望恢复精力并改善工作状态", "patient_goals"),
            )
        ],
    }


def test_l3() -> None:
    """只运行完整 L3：CLINICR、疗法选择和会话级指令编码。"""
    from layers.layer3.api import build_input_contract_report, diagnose

    profile = _fictional_l3_profile()
    blocked = build_input_contract_report(profile)
    if blocked is not None:
        missing = blocked.get("missing_dimensions", [])
        fields = ", ".join(item["field"] for item in missing)
        raise RuntimeError(f"L3 固定画像未通过输入契约：{fields}")
    print("\n[L3 固定虚构患者画像]\n" + _json(profile))
    started = time.perf_counter()
    report = _run_timed(
        "L3 完整诊断",
        lambda: diagnose(
            profile,
            {
                "risk_level": 0,
                "sensitive_topics": [],
                "flags": [],
                "l1_flags": [],
            },
        ),
    )
    elapsed = time.perf_counter() - started
    print("\n[L3 出口：DiagnosticReport 已生成]\n" + _json(report))
    print(f"\n[L3 运行摘要] 耗时约 {elapsed:.2f}s；路由：{report.get('verifier_routing', 'unknown')}")


def _fictional_report():
    """构造通过 L3→L4 接口守卫的虚构报告，不调用 L3。"""
    from layers.layer4.schemas.l3_report import DiagnosticReport
    from shared.report_integrity import REQUIRED_L3_CHECKS, l3_report_fingerprint
    from shared.schemas.treatment_instructions import TreatmentInstructions

    instructions = TreatmentInstructions.model_validate(
        {
            "L1": {
                "identity": "你是一名受过 CBT 训练的心理治疗师",
                "style": "温暖、共情、协作、结构化",
                "boundaries": ["开药", "确诊", "诊断新疾病", "预测未来"],
            },
            "L2": {
                "primary": "MDD",
                "comorbidities": [],
                "excluded": ["Bipolar", "PTSD"],
                "core_beliefs": ["我什么都做不好"],
                "plugins": [],
                "plugin_triggers": [],
                "strategy_adjustments": {
                    "severity_band": "moderate",
                    "motivation_band": "intermediate",
                    "technical_density": "standard",
                    "supportive_response": "standard",
                    "safety_monitoring": "standard",
                    "intervention_focus": ["standard_cbt_seven_steps"],
                    "challenge_interventions": "standard_timing",
                    "exposure_policy": "standard",
                    "care_priority": "psychotherapy",
                    "care_priorities": ["psychotherapy"],
                    "instructions": ["使用标准 CBT 七步流程"],
                    "suppressed_plugins": [],
                },
                "contraindications": [],
                "confidence": 0.86,
                "candidate_confidence": 0.78,
                "verifier_score": 0.86,
                "therapy_type": "CBT+BA",
                "verifier_decision": "confirmed",
                "diagnosis_details": {"primary": {"disorder": "MDD"}},
                "scale_scores": {"PHQ-9": 17},
                "cognitive_patterns": [],
                "triggering_situations": [],
                "key_narratives": [],
                "patient_goals": ["恢复精力并改善工作状态"],
                "allowed_techniques": {
                    "reflection": "开放式提问与共情反映",
                    "feedback": "温和反馈与共同确认",
                    "behavioral_activation": "行为激活",
                },
                "wording_constraint": "使用初步印象措辞，禁止确诊",
            },
            "L3": None,
            "L4": {
                "absolute_bans": ["开药", "确诊", "诊断新疾病", "预测未来"],
                "safety_trigger": "出现自杀意图或计划时立即转交 L1 安全协议",
            },
        }
    )
    report = DiagnosticReport.model_validate(
        {
            "layer_id": "L3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "diagnosis": {
                "primary": {
                    "disorder": "MDD",
                    "confidence": 0.78,
                    "severity": "moderate_severe",
                    "evidence": "虚构量表与叙述证据",
                    "dsm5_criteria_met": ["A1", "A2", "A4", "A6", "A7"],
                    "dsm5_criteria_detail": [
                        {
                            "code": code,
                            "name": name,
                            "status": "met",
                            "met": True,
                            "evidence_refs": refs,
                        }
                        for code, name, refs in (
                            ("A1", "情绪低落", ["PHQ9_item2=3"]),
                            ("A2", "兴趣减退", ["PHQ9_item1=3"]),
                            ("A4", "睡眠障碍", ["PHQ9_item3=2"]),
                            ("A6", "疲劳", ["PHQ9_item4=3"]),
                            ("A7", "无价值感", ["PHQ9_item6=2"]),
                        )
                    ],
                },
                "secondary": [],
                "under_observation": [],
                "ruled_out": [
                    {"disorder": "Bipolar", "reason": "无躁狂或轻躁狂史", "evidence_ref": "MDQ=2"},
                    {"disorder": "PTSD", "reason": "无核心创伤症候群", "evidence_ref": "PCL5=8"},
                ],
            },
            "verifier_score": 0.86,
            "verifier_decision": "confirmed",
            "candidate_confidence": 0.78,
            "verifier_routing": "pass",
            "recommended_therapy": {
                "primary": "CBT+BA",
                "addon": [],
                "plugin_triggers": [],
                "strategy_adjustments": instructions.L2.strategy_adjustments,
                "contraindications": [],
            },
            "severity_level": 2,
            "summary": "虚构测试病例：MDD 初步印象，建议 CBT 联合行为激活。",
            "evidence_refs": [],
            "confidence": 0.86,
            "flags": [],
            "instruction_to_next": "由 L4 按本轮动态指令生成回复。",
            "instructions": instructions.model_dump(mode="json"),
            "l3_validation": {
                "passed": True,
                "route": "allow_l4",
                "checks": [
                    {"rule_id": rule, "status": "pass", "detail": "独立测试固定凭证"}
                    for rule in sorted(REQUIRED_L3_CHECKS)
                ],
                "payload_hash": "pending",
                "policy_version": "L3_OUTPUT_V1",
            },
        }
    )
    report.l3_validation.payload_hash = l3_report_fingerprint(report)
    return report


def _first_turn(instructions, session_id: str, user_message: str):
    from shared.schemas.treatment_instructions import SourceQuote, TurnInstruction

    return TurnInstruction(
        goal="理解患者本轮最主要的困扰并建立合作关系",
        technique_code="reflection",
        technique=instructions.L2.allowed_techniques["reflection"],
        forbidden=list(instructions.L4.absolute_bans),
        link_previous="",
        plugin_guidance=None,
        plugin_steps={},
        force_substage="L4-L1",
        session_id=session_id,
        turn_number=1,
        source_message=user_message,
        evidence=[SourceQuote(source_id="turn:1:user", quote=user_message)],
        observations=[],
        previous_goal_met=False,
        patient_ready=False,
        rationale="患者刚表达当前困扰，首轮应先建立治疗联盟并确认体验。",
    )


def test_l4() -> None:
    """只调用真实 L4 回复模型；使用固定审核器隔离 L5。"""
    from layers.layer4.config import Settings
    from layers.layer4.review.mock_reviewer import MockReviewer
    from layers.layer4.schemas.instructions import SafetyPassport
    from layers.layer4.service import L4Service, SessionRequest, TurnRequest
    from shared.llm_client import get_api_key, get_base_url, get_model
    from shared.schemas.treatment_instructions import TreatmentInstructions

    message = input(
        "患者治疗消息 [这周我还是提不起精神，看到工作消息就想逃避。]："
    ).strip() or "这周我还是提不起精神，看到工作消息就想逃避。"
    report = _fictional_report()
    session_id = "live-l4-" + uuid.uuid4().hex
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compat",
        llm_api_key=get_api_key(),
        llm_base_url=get_base_url(),
        llm_model=get_model(),
        review_mode="mock",
        max_rewrite_attempts=1,
    )
    service = L4Service(config=settings, reviewer=MockReviewer())
    service.create_session(
        SessionRequest(
            session_id=session_id,
            report=report,
            safety=SafetyPassport(risk_level=0, sensitive_topics=[], flags=[]),
        )
    )
    instructions = TreatmentInstructions.model_validate(report.instructions)
    turn = _first_turn(instructions, session_id, message)
    outcome = _run_timed(
        "L4 治疗回复",
        lambda: service.handle_turn(
            TurnRequest(session_id=session_id, user_message=message, turn=turn)
        ),
    )
    print("\n[L4 出口：本轮治疗回复已生成]\n" + outcome.reply)
    print("\n[L4 结构化结果]\n" + _json(outcome))
    print("\n说明：本项只隔离测试 L4 的真实生成模型，L5 审核未被调用。")


def test_l5() -> None:
    """只调用真实 L5 审核模型；L4 回复与完整上下文由固定样例提供。"""
    from layers.layer4.generation.summary_builder import build_l5_context
    from layers.layer4.schemas.instructions import SafetyPassport
    from layers.layer5.reviewer import review_response
    from shared.schemas.treatment_instructions import TreatmentInstructions

    user_message = input(
        "本轮患者消息 [这周我还是提不起精神，看到工作消息就想逃避。]："
    ).strip() or "这周我还是提不起精神，看到工作消息就想逃避。"
    l4_reply = input(
        "待审核 L4 回复 [听起来工作消息会立刻带来很强的压力。我们可以先看看那一刻脑中出现了什么想法。]："
    ).strip() or "听起来工作消息会立刻带来很强的压力。我们可以先看看那一刻脑中出现了什么想法。"
    report = _fictional_report()
    session_id = "live-l5-" + uuid.uuid4().hex
    instructions = TreatmentInstructions.model_validate(report.instructions)
    instructions = instructions.model_copy(
        update={"L3": _first_turn(instructions, session_id, user_message)},
        deep=True,
    )
    context = build_l5_context(
        l4_reply,
        "L4-L1",
        SafetyPassport(risk_level=0, sensitive_topics=[], flags=[]),
        instructions,
        user_message,
        [],
    )
    result = _run_timed("L5 输出审核", lambda: review_response(context))
    print("\n[L5 出口：本轮审核判定已生成]\n" + _json(result))


TESTS: dict[str, tuple[str, Callable[[], None]]] = {
    "1": ("L1 安全层", test_l1),
    "2": ("L2 评估层", test_l2),
    "3": ("L3 诊断层", test_l3),
    "4": ("L4 治疗层", test_l4),
    "5": ("L5 审核层", test_l5),
}
ALIASES = {f"l{number}": number for number in TESTS}


def _choose_layer() -> str:
    print("\n请选择要独立测试的层级：")
    for number, (name, _) in TESTS.items():
        print(f"  {number}. {name}")
    print("  0. 退出")
    return input("选择：").strip().lower()


def _execute(choice: str) -> bool:
    choice = ALIASES.get(choice, choice)
    if choice in {"0", "exit", "quit", "q"}:
        return False
    if choice not in TESTS:
        print("无效选择，请输入 1–5。")
        return True
    name, test = TESTS[choice]
    print(f"\n=== {name}：独立真实 LLM 测试 ===")
    try:
        _configure_real_api()
        test()
    except Exception as exc:
        if not getattr(exc, "_live_error_shown", False):
            _show_safe_exception_chain(exc)
        print("本层测试未通过；上方安全错误链用于定位，不会继续调用其他层。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="L1–L5 独立真实 LLM 手工测试")
    parser.add_argument(
        "layer",
        nargs="?",
        choices=["l1", "l2", "l3", "l4", "l5"],
        help="直接运行指定层；省略时显示交互菜单",
    )
    args = parser.parse_args()
    print("五层系统 · 独立真实 LLM 测试")
    print("每次只运行所选层；请求会发送到配置的模型服务并可能产生费用。")
    if args.layer:
        _execute(args.layer)
        return 0
    while _execute(_choose_layer()):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
