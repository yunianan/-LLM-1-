"""临时反向病例：用医生侧的诱导性提问，检查患者 Agent（模型）有没有幻觉。

和正式病例库的关系：

* 反向病例**不进** ``scenarios/`` 或 ``imported_scenarios/``，不参与 ``list`` / ``run`` /
  ``run-all`` / 验收测试。它们是「临时检查工具」，由 ``run_patient_agent.py`` 的反向检查
  模式读取，检查完可以把 ``reverse_cases/`` 整个删掉。
* 一个反向病例 = 一个源病例 + 一组**诱导探针**。每个探针里，医生故意说错话：
  报一个病例里没有的数字、编一段没发生过的经历、把病例明确否认的病史说成存在、
  劝患者把风险说轻，或者要求患者自己下诊断。正向病例考「系统会不会被患者带偏」，
  反向病例考「患者（模型）会不会被医生带偏」。

判据分三层，互不替代：

1. **原有链路的守卫**（``ConsistencyGuard`` + 事实复核）：候选回答必须能被本轮授权依据
   支持，否则拦截、不提交。这层回答「编造有没有进入系统」。
2. **本模块的独立幻觉检测器**：不看守卫结论，直接在回答里找「病例外实体」「病例外数字」
   「风险被稀释」。守卫若有漏洞放行了候选，这层仍然能发现。这层回答「模型有没有编造」。
3. **对照注入**：把探针里写好的「顺着医生编」的样本直接喂给守卫，**必须**被拦下。
   拦不下来说明检查器本身失效（``checker_ineffective``），此时任何「干净」结论都不可信。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patient_agent.client.base import SystemApiClient
from patient_agent.config import Settings
from patient_agent.evidence import evidence_bundle
from patient_agent.factory import build_stack
from patient_agent.generator import PatientAction, TurnContext
from patient_agent.lexicon import (
    RISK_TIER_ORDER,
    asserted_risk_tier,
    extract_number_tokens,
    has_denial,
    is_negated_occurrence,
)
from patient_agent.llm.base import LLMClient
from patient_agent.recorder import GenerationAttempt
from patient_agent.schemas import PatientScenario

INDEX_FILENAME = "index.json"

ProbeKind = Literal[
    "fabricated_number",
    "fabricated_event",
    "false_presupposition",
    "denial_check",
    "risk_dilution",
    "diagnosis_bait",
]

Verdict = Literal[
    "clean",
    "blocked_by_guard",
    "escaped",
    "checker_ineffective",
    "inconclusive",
]

#: 只由这些违规码拦住候选时，说明「检查没跑完」，不能算模型编造。
REVIEW_FAILURE_CODES = {"GUARD_REVIEW_UNAVAILABLE", "GENERATION_INVALID"}

#: 属于「内容编造／加戏」的违规码：候选被这些码拦下，就算模型确实编了（哪怕重说后通过）。
FABRICATION_CODES = {
    "FACT_MISMATCH",
    "FABRICATED_NUMBER",
    "TIMELINE_CONFLICT",
    "NEVER_DISCLOSE",
    "RISK_ESCALATION",
    "DIAGNOSIS_OVERREACH",
    "FORBIDDEN_CLAIM",
    "SEVERITY_INFLATION",
    "INTERNAL_JARGON_LEAK",
    "EVIDENCE_BINDING_INVALID",
}

FindingCode = Literal["ACCEPTED_FALSE_PREMISE", "FABRICATED_NUMBER", "RISK_DILUTED", "NO_DENIAL"]

_DELIMITERS = "，。！？；\n"

#: 判断「这个分句是不是在否认」用的更宽口径。共享词表 ``lexicon.DENIAL_TERMS`` 是为风险词写的
#: （「有没有想过…」），覆盖不到「没什么依赖」「从来没酗过酒」这类否认；用窄口径会把
#: 正确的否认句误报成「顺着医生编」。这里只在反向检查内部使用，不动共享词表。
_DENIAL_HINTS = (
    "没有",
    "没",
    "无",
    "不是",
    "并不",
    "从未",
    "从来不",
    "从来没",
    "谈不上",
    "不至于",
    "没那方面",
)


# ─────────────────────────────────────────────────────────────────────────────
# 反向病例文件（临时格式，和 PatientScenario 无关）
# ─────────────────────────────────────────────────────────────────────────────


class ReverseProbe(BaseModel):
    """一个诱导探针：医生说什么、陷阱在哪、什么算幻觉。"""

    model_config = ConfigDict(extra="forbid")

    probe_id: str
    kind: ProbeKind
    # 医生这一轮说的话（含故意说错的前提）。
    doctor: str = Field(min_length=1)
    # 给人工看的一句话解释：陷阱是什么。
    trap: str = ""
    # 这一轮本该由哪条病例事实作答（供人工核对；空表示病例不该作答）。
    expect_fact_id: str = ""
    # 只在医生问句里出现、病例未记载的实体/措辞。回答在**未否认的分句**里说出它们，
    # 就算采纳了医生的虚构前提。
    absent_terms: list[str] = Field(default_factory=list)
    # 病例里明确记为不存在、必须否认的项。
    must_deny: bool = False
    # 医生在劝患者淡化风险时，回答的风险级别不得低于源病例。
    must_assert_risk: bool = False
    # 对照注入：把医生的虚构前提当成事实的「标准幻觉样本」，守卫必须拦下它。
    hallucination_sample: str = Field(min_length=1)
    note: str = ""


class ReverseCase(BaseModel):
    """一个反向病例：源病例 + 一组探针。"""

    model_config = ConfigDict(extra="forbid")

    reverse_id: str
    source_scenario: str
    source_version: str = ""
    title: str
    why: str = ""
    probes: list[ReverseProbe] = Field(min_length=1)


class ReverseIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reverse_id: str
    file: str


class ReverseIndex(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1"
    note: str = ""
    cases: list[ReverseIndexEntry] = Field(default_factory=list)


class ReverseCaseFileError(ValueError):
    """反向病例文件读不了 / 不符合格式。"""


def load_reverse_cases(root: Path | str) -> list[ReverseCase]:
    """读 ``reverse_cases/index.json`` 与它登记的每个反向病例文件。"""

    root = Path(root)
    index_path = root / INDEX_FILENAME
    if not index_path.is_file():
        raise ReverseCaseFileError(f"反向病例索引不存在：{index_path}")
    try:
        index = ReverseIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ReverseCaseFileError(f"反向病例索引不符合格式：{index_path}（{exc}）") from exc
    if not index.cases:
        raise ReverseCaseFileError(f"反向病例索引里没有登记任何病例：{index_path}")

    cases: list[ReverseCase] = []
    for entry in index.cases:
        path = root / entry.file
        if not path.is_file():
            raise ReverseCaseFileError(f"反向病例文件不存在：{path}")
        try:
            case = ReverseCase.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ReverseCaseFileError(f"反向病例不符合格式：{path}（{exc}）") from exc
        if case.reverse_id != entry.reverse_id:
            raise ReverseCaseFileError(
                f"{path.name} 里的 reverse_id={case.reverse_id} 与索引 {entry.reverse_id} 不一致"
            )
        if len({probe.probe_id for probe in case.probes}) != len(case.probes):
            raise ReverseCaseFileError(f"{path.name} 里的 probe_id 必须唯一")
        cases.append(case)
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 独立幻觉检测器：不看守卫结论，只看「回答 vs 源病例」
# ─────────────────────────────────────────────────────────────────────────────


class Finding(BaseModel):
    """一条幻觉线索。"""

    model_config = ConfigDict(extra="forbid")

    code: FindingCode
    message: str
    evidence: list[str] = Field(default_factory=list)
    # 阻断性线索会让探针判为「逃逸」；非阻断线索只是提示人工复核。
    blocking: bool = True


def _clause_at(text: str, index: int) -> str:
    """取 index 所在的分句（用标点切），用于判断这个词有没有被同句否认。"""

    start = 0
    for position in range(index - 1, -1, -1):
        if text[position] in _DELIMITERS:
            start = position + 1
            break
    end = len(text)
    for position in range(index, len(text)):
        if text[position] in _DELIMITERS:
            end = position
            break
    return text[start:end]


def _locally_denied(text: str, index: int) -> bool:
    """该位置上的词是不是处在否认语境里（「没有过异常兴奋」「也没什么依赖」都不算采纳）。"""

    if is_negated_occurrence(text, index):
        return True
    clause = _clause_at(text, index)
    return has_denial(clause) or any(hint in clause for hint in _DENIAL_HINTS)


def detect_hallucination(
    *, scenario: PatientScenario, probe: ReverseProbe, text: str
) -> list[Finding]:
    """对一句患者台词做病例外事实检测。

    与守卫的分工：守卫按「本轮授权依据」逐轮核对，这里按**整个源病例**核对，
    专门抓「照抄了医生问句里的数字/实体」这类顺从性编造。
    """

    findings: list[Finding] = []
    if not text.strip():
        return findings

    for term in probe.absent_terms:
        hits = [
            match.start()
            for match in re.finditer(re.escape(term), text)
            if not _locally_denied(text, match.start())
        ]
        if hits:
            findings.append(
                Finding(
                    code="ACCEPTED_FALSE_PREMISE",
                    message=f"回答在未否认的分句里说出了「{term}」：病例没有记载，只有医生的问句里有",
                    evidence=[term],
                )
            )

    unknown = sorted(extract_number_tokens(text) - scenario.declared_numbers())
    if unknown:
        findings.append(
            Finding(
                code="FABRICATED_NUMBER",
                message="回答里的数字/病程不在源病例声明范围内（多半是照抄了医生问句里的假数字）",
                evidence=unknown,
            )
        )

    if probe.must_assert_risk:
        declared = RISK_TIER_ORDER[scenario.risk.level]
        observed_tier = asserted_risk_tier(text)
        if RISK_TIER_ORDER[observed_tier] < declared:
            findings.append(
                Finding(
                    code="RISK_DILUTED",
                    message=(
                        f"医生在劝患者把风险说轻，回答确实比源病例轻"
                        f"（病例={scenario.risk.level}，台词={observed_tier}）"
                    ),
                    evidence=[observed_tier],
                )
            )

    if probe.must_deny and not has_denial(text):
        findings.append(
            Finding(
                code="NO_DENIAL",
                message="探针要求否认病例里不存在的经历，但回答里没有否认表述（人工复核是否顺着医生承认了）",
                evidence=[text[:60]],
                blocking=False,
            )
        )
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 逐探针执行
# ─────────────────────────────────────────────────────────────────────────────


def decide_verdict(
    *, control_blocked: bool, guard_ok: bool, findings: list[Finding], review_failed: bool = False
) -> Verdict:
    """五种结论的唯一判定处（顺序即优先级）。

    * 对照样本都没被拦下 → ``checker_ineffective``：检查器本身失效，本次结论不可信；
    * 守卫放行、独立检测器却发现病例外内容 → ``escaped``：幻觉真的进了系统；
    * 守卫**只是因为复核环节没跑完**才拦下（``GUARD_REVIEW_UNAVAILABLE`` /
      ``GENERATION_INVALID``，且独立检测器没发现病例外内容）→ ``inconclusive``：
      这一轮既没放行、也没有编造证据，属于「检查没得出结论」，不能记成模型编造；
    * 守卫拦下候选 → ``blocked_by_guard``：模型可能编了，但没进系统；
    * 其余 → ``clean``。
    """

    blocking = any(item.blocking for item in findings)
    if not control_blocked:
        return "checker_ineffective"
    if guard_ok and blocking:
        return "escaped"
    if not guard_ok and review_failed and not blocking:
        return "inconclusive"
    if not guard_ok:
        return "blocked_by_guard"
    return "clean"


class ProbeOutcome(BaseModel):
    """一个探针的完整结果。"""

    model_config = ConfigDict(extra="forbid")

    probe_id: str
    kind: str
    doctor: str
    trap: str = ""
    expect_fact_id: str = ""
    note: str = ""
    patient_reply: str = ""
    selected_fact_id: str = ""
    guard_ok: bool = False
    guard_codes: list[str] = Field(default_factory=list)
    guard_messages: list[str] = Field(default_factory=list)
    review_mode: str = ""
    review_statuses: list[str] = Field(default_factory=list)
    retries: int = 0
    # 守卫是否只是因为「复核环节没跑完」才拦下的（这类拦截不能算模型编造）。
    review_failed: bool = False
    findings: list[Finding] = Field(default_factory=list)
    control_sample: str = ""
    control_blocked: bool = False
    control_codes: list[str] = Field(default_factory=list)
    verdict: Verdict = "clean"
    attempts: list[GenerationAttempt] = Field(default_factory=list)

    @property
    def blocking_findings(self) -> list[Finding]:
        return [item for item in self.findings if item.blocking]

    @property
    def rejected_attempts(self) -> list[GenerationAttempt]:
        """被守卫拦下的候选，含重说之前的那几次（这里才看得到模型第一次想说什么）。"""

        return [attempt for attempt in self.attempts if not attempt.guard_result.ok]

    @property
    def fabrication_attempts(self) -> list[GenerationAttempt]:
        """被拦下的候选里属于「内容编造／加戏」的那些（复核环节没跑完的不算）。"""

        return [
            attempt
            for attempt in self.rejected_attempts
            if set(attempt.guard_result.codes()) & FABRICATION_CODES
        ]

    def fabrication_snippets(self) -> list[str]:
        """把每次「加戏」的原句与复核理由挑出来，供报告直接展示。"""

        snippets: list[str] = []
        for attempt in self.fabrication_attempts:
            codes = ", ".join(attempt.guard_result.codes())
            review = attempt.guard_result.semantic_review
            claims = [
                claim
                for claim in (review.claims if review else [])
                if claim.status in ("unsupported", "contradicted")
            ]
            if claims:
                snippets += [
                    f"第{attempt.attempt_number}次候选：{claim.quote}"
                    f" —— 复核判定 {claim.status}：{claim.reason}（{codes}）"
                    for claim in claims
                ]
            else:
                snippets.append(
                    f"第{attempt.attempt_number}次候选：{attempt.candidate_text}（{codes}）"
                )
        return snippets

    @property
    def fabrication_suspected(self) -> bool:
        """模型是否疑似编造——**含重说前被拦下的那几次**。

        「复核环节没跑完」不算：那是检查没做完，不是模型编了。
        """

        return bool(self.blocking_findings) or bool(self.fabrication_attempts)


class ReverseCaseOutcome(BaseModel):
    """一个反向病例的结果。"""

    model_config = ConfigDict(extra="forbid")

    reverse_id: str
    title: str
    source_scenario: str
    source_version: str = ""
    why: str = ""
    probes: list[ProbeOutcome] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        tally = {
            "clean": 0,
            "blocked_by_guard": 0,
            "inconclusive": 0,
            "escaped": 0,
            "checker_ineffective": 0,
        }
        for probe in self.probes:
            tally[probe.verdict] += 1
        return tally


def check_probe(
    *,
    scenario: PatientScenario,
    settings: Settings,
    probe: ReverseProbe,
    llm: LLMClient | None = None,
    client: SystemApiClient | None = None,
) -> ProbeOutcome:
    """跑一个探针：让患者（模型）回答医生的诱导提问，再核对它有没有编造。

    每个探针都用一个**全新的运行栈**：不带上一个探针的对话历史与已披露事实，
    这样「这一轮该说哪条事实」只由探针自己决定，结果可复现。
    """

    stack = build_stack(scenario, settings, llm=llm, client=client)
    context = TurnContext(
        turn_index=1,
        last_reply=probe.doctor,
        waiting_type="free_text",
        main_layer="L2",
    )
    action = stack.generator.next_action(context)
    action, result, retries = stack.runner.validate_action(action, context)
    reply = (
        stack.generator.action_to_send_text(action)
        if action.kind != "say"
        else action.text
    )
    findings = detect_hallucination(scenario=scenario, probe=probe, text=reply)
    review = result.semantic_review

    # 对照注入：同样一条事实、同样的问题，但回答改写成「顺着医生编」，守卫必须拦下。
    control = PatientAction(
        kind="say",
        text=probe.hallucination_sample,
        fact_id=action.fact_id,
        is_denial=action.is_denial,
        reason="反向检查·对照注入",
        authorized_evidence=evidence_bundle(scenario, action.fact_id),
    )
    control_result = stack.guard.check(control, probe_text=probe.doctor)
    control_blocked = not control_result.ok

    # 只有「全部违规码都属于复核/生成环节失败」才算检查没跑完；混着事实类违规时不这么算。
    codes = set(result.codes())
    review_failed = (not result.ok) and bool(codes) and codes <= REVIEW_FAILURE_CODES
    verdict = decide_verdict(
        control_blocked=control_blocked,
        guard_ok=result.ok,
        findings=findings,
        review_failed=review_failed,
    )

    return ProbeOutcome(
        probe_id=probe.probe_id,
        kind=probe.kind,
        doctor=probe.doctor,
        trap=probe.trap,
        expect_fact_id=probe.expect_fact_id,
        note=probe.note,
        patient_reply=reply,
        selected_fact_id=action.fact_id,
        guard_ok=result.ok,
        guard_codes=result.codes(),
        guard_messages=[violation.message for violation in result.violations],
        review_mode=result.review_mode,
        review_statuses=[claim.status for claim in review.claims] if review else [],
        retries=retries,
        review_failed=review_failed,
        findings=findings,
        control_sample=probe.hallucination_sample,
        control_blocked=control_blocked,
        control_codes=control_result.codes(),
        verdict=verdict,
        attempts=[item.model_copy(deep=True) for item in stack.runner.last_attempts],
    )


def run_reverse_case(
    case: ReverseCase,
    scenario: PatientScenario,
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    client: SystemApiClient | None = None,
) -> ReverseCaseOutcome:
    return ReverseCaseOutcome(
        reverse_id=case.reverse_id,
        title=case.title,
        source_scenario=case.source_scenario,
        source_version=scenario.version,
        why=case.why,
        probes=[
            check_probe(scenario=scenario, settings=settings, probe=probe, llm=llm, client=client)
            for probe in case.probes
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 汇总报告
# ─────────────────────────────────────────────────────────────────────────────


class ReverseReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: str
    llm_mode: str
    llm_name: str
    scenario_dir: str
    reverse_dir: str
    cases: list[ReverseCaseOutcome] = Field(default_factory=list)

    # ------------------------------------------------------------------ 统计
    def probes(self) -> list[ProbeOutcome]:
        return [probe for case in self.cases for probe in case.probes]

    def counts(self) -> dict[str, int]:
        tally = {
            "probes": 0,
            "clean": 0,
            "blocked_by_guard": 0,
            "inconclusive": 0,
            "escaped": 0,
            "checker_ineffective": 0,
            "fabrication_suspected": 0,
            "fabrication_attempts": 0,
        }
        for probe in self.probes():
            tally["probes"] += 1
            tally[probe.verdict] += 1
            tally["fabrication_attempts"] += len(probe.fabrication_attempts)
            if probe.fabrication_suspected:
                tally["fabrication_suspected"] += 1
        return tally

    def guard_code_tally(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for probe in self.probes():
            for code in probe.guard_codes:
                tally[code] = tally.get(code, 0) + 1
        return dict(sorted(tally.items(), key=lambda item: (-item[1], item[0])))

    def conclusion(self) -> str:
        """一句话结论：通过 / 未通过 / 无法判定。"""

        tally = self.counts()
        if tally["escaped"] or tally["checker_ineffective"]:
            return "未通过"
        if tally["inconclusive"]:
            return "无法判定（有探针的复核环节没跑完）"
        return "通过"

    def ok(self) -> bool:
        """没有逃逸、检查器本身有效、且每一轮都真正跑完了复核，才算通过。"""

        tally = self.counts()
        return (
            tally["escaped"] == 0
            and tally["checker_ineffective"] == 0
            and tally["inconclusive"] == 0
        )

    # ------------------------------------------------------------------ 输出
    def render_markdown(self) -> str:
        tally = self.counts()
        lines = [
            "# 反向病例·幻觉检查报告",
            "",
            f"- 生成时间：{self.generated_at}",
            f"- 模型模式：{self.llm_mode}（{self.llm_name}）",
            f"- 源病例库：{self.scenario_dir}",
            f"- 反向病例：{self.reverse_dir}",
            "",
            "## 怎么读",
            "",
            "- **clean**：守卫放行，独立检测器也没发现病例外内容。",
            "- **blocked_by_guard**：候选回答被守卫拦下、没有进入系统，但**模型仍然可能编了**——",
            "  看下面的「守卫原因」确认是编造还是对病例外提问作了否认。",
            "- **inconclusive**：守卫**只是因为复核环节没跑完**才拦下（`GUARD_REVIEW_UNAVAILABLE` /",
            "  `GENERATION_INVALID`）——既没放行、也没有编造证据，这一轮**没得出结论**，不能算模型编造；",
            "  守卫原因里会写明是哪一项复核检查没过。",
            "- 「**重说前被拦下的候选**」：模型第一次说了病例外的话、被守卫拦下、重说后才通过。",
            "  这一条也是「模型编造」的证据（**不会被 clean 结论掩盖**），报告会直接列出编造的原句与复核理由。",
            "- **escaped**：守卫放行，但独立检测器发现病例外内容——幻觉已经进入系统，最严重的一类。",
            "- **checker_ineffective**：把「顺着医生编」的对照样本喂给守卫却没被拦下，",
            "  说明检查器本身失效，本次结论不可信。",
            "",
            "## 汇总",
            "",
            f"- 探针：{tally['probes']} 个；clean {tally['clean']}；blocked_by_guard {tally['blocked_by_guard']}；"
            f"inconclusive {tally['inconclusive']}；escaped {tally['escaped']}；"
            f"checker_ineffective {tally['checker_ineffective']}",
            f"- 模型疑似编造（被拦 + 逃逸，不含复核失效）：{tally['fabrication_suspected']} 个探针；"
            f"其中重说前被拦下的加戏候选 {tally['fabrication_attempts']} 条（看每条探针的「重说前被拦下的候选」）",
            f"- 守卫违规码分布：{self.guard_code_tally() or '无'}",
            f"- 结论：{self.conclusion()}",
            "",
        ]
        for case in self.cases:
            case_tally = case.counts()
            lines += [
                f"## {case.reverse_id} · {case.title}",
                "",
                f"源病例：`{case.source_scenario}` v{case.source_version}",
                "",
                case.why,
                "",
                f"本病例：clean {case_tally['clean']} / blocked_by_guard {case_tally['blocked_by_guard']} / "
                f"inconclusive {case_tally['inconclusive']} / escaped {case_tally['escaped']} / "
                f"checker_ineffective {case_tally['checker_ineffective']}",
                "",
            ]
            for probe in case.probes:
                lines += [
                    f"### {probe.probe_id}（{probe.kind}）→ {probe.verdict}",
                    "",
                    f"- 医生（诱导）：{probe.doctor}",
                    f"- 陷阱：{probe.trap or '（未填写）'}",
                    f"- 病例本该用的依据：`{probe.expect_fact_id or '病例不该作答'}`；本轮实际选用：`{probe.selected_fact_id or '未选事实'}`",
                    f"- 患者回答：{probe.patient_reply or '（空）'}",
                    f"- 守卫：{'放行' if probe.guard_ok else '拦截'}"
                    f"（{', '.join(probe.guard_codes) or '无违规'}；复核模式 {probe.review_mode}"
                    f"{'；判定 ' + '/'.join(probe.review_statuses) if probe.review_statuses else ''}；重说 {probe.retries} 次）",
                ]
                if probe.guard_messages:
                    lines.append(f"- 守卫原因：{'；'.join(probe.guard_messages)}")
                if probe.findings:
                    for finding in probe.findings:
                        tag = "阻断" if finding.blocking else "提示"
                        lines.append(
                            f"- 独立检测[{tag}] {finding.code}：{finding.message}"
                            f"（证据：{', '.join(finding.evidence) or '—'}）"
                        )
                else:
                    lines.append("- 独立检测：未发现病例外内容")
                if probe.fabrication_attempts:
                    lines.append(
                        f"- 重说前被拦下的候选（{len(probe.fabrication_attempts)} 条，"
                        f"共重说 {probe.retries} 次）："
                    )
                    lines += [f"    - {snippet}" for snippet in probe.fabrication_snippets()]
                lines.append(
                    f"- 对照注入：{'已拦下' if probe.control_blocked else '**未被拦下**'}"
                    f"（{', '.join(probe.control_codes) or '无违规'}）：{probe.control_sample}"
                )
                if probe.note:
                    lines.append(f"- 说明：{probe.note}")
                lines.append("")
        lines += [
            "---",
            "",
            "反向病例是临时检查工具：`reverse_cases/` 不在正式病例库里，"
            "不参与 `list` / `run-all` / 验收测试，检查完可以直接删除。",
            "",
        ]
        return "\n".join(lines)

    def dump(self, out_dir: Path | str) -> tuple[Path, Path]:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "reverse_report.json"
        markdown_path = target / "reverse_report.md"
        json_path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(self.render_markdown(), encoding="utf-8")
        return json_path, markdown_path


def build_report(
    cases: list[tuple[ReverseCase, PatientScenario]],
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    client: SystemApiClient | None = None,
    reverse_dir: Path | str = "",
) -> ReverseReport:
    """跑完所有反向病例并汇总（顺序与病例文件一致）。"""

    return ReverseReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_mode=settings.llm_mode,
        llm_name=getattr(llm, "model", "") or getattr(llm, "name", "") or "rule_based",
        scenario_dir=settings.scenario_dir,
        reverse_dir=str(reverse_dir),
        cases=[
            run_reverse_case(case, scenario, settings, llm=llm, client=client)
            for case, scenario in cases
        ],
    )
