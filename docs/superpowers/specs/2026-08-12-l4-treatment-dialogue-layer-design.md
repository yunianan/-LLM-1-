# L4 治疗对话层（Treatment Dialogue Layer）设计文档

日期：2026-08-12
状态：已确认

## 1. 背景与定位

本系统采用五层认知架构（Rollwage et al., 2026, Nature Medicine 认知层架构的扩展），将临床推理与语言生成分离：

```
Layer1 安全评估 → Layer2 MI评估 → Layer3 诊断推理(CLINICR) → Layer4 治疗对话 → Layer5 输出安全审核
```

L4 治疗对话层的核心职责：**接收 L3 诊断层下发的 L1-L4 治疗指令，将其"翻译"为自然、温暖、专业的治疗对话，直接与患者交互。本层不做任何临床推理。**

- 诊断层是"大脑"，L4 是"嘴巴"：大脑决定"现在应该做什么"，嘴巴决定"这句话怎么说才合适"。
- L4 只决定语气、措辞、节奏，不参与治疗方案的推理。
- 层内模型现阶段临时调用 API（OpenAI 兼容接口，如 DeepSeek），后期可无缝切换到本地训练的模型，不需要训练模型。

## 2. 需求来源

依据两份文档：

1. 《心理诊疗LLM系统五层认知架构》（项目架构说明）
   - L4 定位与职责（第四层治疗对话层）
   - 输入输出定义（第三层 L1-L4 分层治疗指令生成、第四层输入与输出）
   - CBT 标准七步会话结构
   - 插件激活时的变化
   - L4 的 prompt 组装
   - 接口三 L3→L4 DiagnosticReport（L4 输入）
   - 接口四 L4+L3→L5 ContextualSafetyContext（L4 输出）
   - 结构化临床摘要通用格式
   - 回环A（层内迭代）与回环B（跨层回溯）触发条件
   - 接口校验规则 V4-1 ~ V4-4
   - 跨层反思检验框架中 L4 层内自查指标
2. Rollwage et al. (2026, Nature Medicine)《A cognitive layer architecture to support large-language model performance in psychotherapy interactions》
   - 输入层/输出层双向架构、动态 prompt 注入机制
   - 会话按阶段模块化（agenda setting / information collection / formulation / intervention selection / intervention delivery / wrap-up），每个阶段有完成判定后推进

## 3. 需求澄清（与用户确认的决策）

| 决策点 | 选择 |
|---|---|
| 交付形态 | 完整可运行代码 |
| 相邻层（L1/L3/L5）处理 | 定义接口数据契约 + Mock/Stub 实现，后期真实层接入直接替换 |
| 技术栈 | Python 3.11+ / Pydantic v2 / FastAPI |
| LLM 接入 | OpenAI 兼容接口（base_url 可配 DeepSeek 等）+ MockBackend；预留 LocalModelBackend 占位以接入本地训练模型 |
| L4 子层状态机 | 完整状态机 L4-L1→L4-L2→L4-L3→L4-L4，跨轮推进，L3 指令可覆盖阶段 |
| 回环A 编排 | L4 内部编排（生成→审核→重写≤3轮），审核器通过协议注入，L5 就绪后替换 |
| 测试验证 | pytest 单测 + CLI 演示脚本（MockBackend 无密钥可跑） |

## 4. 模块划分（端口-适配器分层）

```
Layer 4/
├── pyproject.toml
├── README.md
├── .env.example
├── l4/
│   ├── config.py
│   ├── schemas/
│   │   ├── instructions.py   # 输入契约：L1角色/L2诊断锚点/L3动态指令/L4防越界
│   │   ├── l3_report.py      # 接口三 DiagnosticReport（L3→L4）
│   │   ├── l5_context.py     # 接口四 ContextualSafetyContext（L4→L5）
│   │   ├── summary.py        # 结构化临床摘要通用格式 + evidence_refs
│   │   └── review.py         # 审核契约 ReviewResult
│   ├── llm/
│   │   ├── base.py           # LLMBackend 协议
│   │   ├── openai_compat.py  # OpenAI 兼容适配器（DeepSeek）
│   │   ├── mock.py           # FakeLLM
│   │   └── local.py          # 本地模型适配器占位
│   ├── session/
│   │   ├── state_machine.py  # L4 子层状态机
│   │   └── history.py        # 会话历史与元数据
│   ├── generation/
│   │   ├── prompt_assembler.py
│   │   ├── responder.py
│   │   └── summary_builder.py
│   ├── review/
│   │   ├── reviewer.py       # 审核端口协议
│   │   ├── mock_reviewer.py  # Mock 审核器
│   │   └── orchestrator.py   # 回环A 编排
│   ├── service.py            # L4 门面
│   ├── api/server.py         # FastAPI 服务
│   └── cli/demo.py           # CLI 演示
├── tests/                    # pytest 单测
└── examples/
    └── sample_diagnostic_report.json   # 示例 L3 输入
```

## 5. 数据契约（schemas）

全部使用 Pydantic v2 模型，字段与架构文档 JSON 示例一一对应。

### 5.1 输入契约 `instructions.py`

- `RoleInstruction`（L1 角色定义，全会话复用）：`identity`, `style`
- `DiagnosticAnchor`（L2 诊断锚点，全会话复用）：`primary`, `comorbidities`, `excluded`, `core_beliefs`, `plugins`, `confidence`
- `TurnInstruction`（L3 本轮动态指令，每轮重生成）：`goal`, `technique`, `forbidden`, `link_previous`, `plugin_guidance`, `force_substage`
- `BoundaryInstruction`（L4 防越界，每轮复用）：`absolute_bans`, `safety_trigger`

### 5.2 接口三 `l3_report.py`：DiagnosticReport（L3→L4）

完整对应文档示例字段：`diagnosis.primary/secondary/ruled_out`、`verifier_score`、`verifier_routing`、`recommended_therapy(primary/addon/contraindications)`、`severity_level`、`clinical_notes_for_L4`、`summary`、`structured_labels.therapy_routing(l4_substage_priority)`、`structured_labels.wording_constraint`、`evidence_refs`、`confidence`、`flags`、`instruction_to_next`。

### 5.3 接口四 `l5_context.py`：ContextualSafetyContext（L4→L5）

```python
class L5Context(BaseModel):
    l4_raw_output: str
    l4_substage: Literal["L4-L1", "L4-L2", "L4-L3", "L4-L4"]  # V4-1 枚举
    therapy_type: str                                          # V4-3 匹配
    diagnosis_context: DiagnosisContext                        # primary/secondary/suicidal_ideation
    risk_context: RiskContext                                  # risk_level/sensitive_topics
```

### 5.4 结构化临床摘要 `summary.py`

通用格式：`layer_id="L4"`, `timestamp`, `summary`, `structured_labels`, `evidence_refs[source/finding/basis]`, `confidence`, `flags`。

`evidence_refs` 必须包含：L3 指令引用、对话历史片段、L1 安全标记，支撑第三级端到端逆溯检验。

### 5.5 审核契约 `review.py`

`ReviewResult`: `harm/boundary/quality: pass|fail`, `fail_reason`, `fix_instruction`。Mock 审核器与真实 L5 共用。

## 6. LLM 后端抽象（llm）

### 6.1 端口协议 `base.py`

```python
class LLMBackend(Protocol):
    def chat(self, messages: list[Message], *, temperature: float = 0.7,
             json_mode: bool = False) -> str: ...
```

### 6.2 适配器

| 适配器 | 说明 |
|---|---|
| `OpenAICompatBackend` | `openai` 客户端，`base_url` 可配（DeepSeek: `https://api.deepseek.com`），支持 `json_mode` |
| `MockBackend` | 脚本化回复（关键词匹配预设回复），无密钥可跑测试/演示 |
| `LocalModelBackend` | 占位骨架（vLLM/transformers 接入点 TODO），后期填充不改变其他代码 |

### 6.3 工厂

`get_backend(config)` 按 `config.llm_provider` 返回实例，运行时可在 API/本地模型间切换，体现模型无关性。

## 7. 会话状态机（session）

### 7.1 状态定义（对应 l4_substage 枚举）

```
L4-L1 共情建立（Rapport）      → 建立联盟、不深挖创伤
L4-L2 结果反馈（Feedback）     → 温和反馈"初步印象"、征求患者确认
L4-L3 技术执行（Intervention） → 按 L3 指定技术推进核心工作
L4-L4 作业布置（Homework）     → 布置家庭作业 + 总结 + 反馈
```

### 7.2 推进规则（三路信号取最高优先级）

1. **L3 强制覆盖（最高）**：`TurnInstruction.force_substage` 非空直接跳转（诊断层权威）。
2. **阶段完成信号（默认）**：L3 未覆盖时按 L4-L1→L4-L2→L4-L3→L4-L4 顺序推进。每次进入新阶段调用 LLM（json_mode）判定"本轮目标是否达成"，达成后下一轮进入下一阶段。
3. **闭环保护**：只允许前向推进；L4-L4 为终态；L3 覆盖导致的异常阶段序列记录到 `flags`。

### 7.3 历史存储 `history.py`

按 `session_id` 维护 `conversation_history`、当前 `l4_substage`、已执行 L3 指令列表、重试计数，供 prompt 组装与 evidence_refs 引用。

## 8. Prompt 组装与生成（generation）

### 8.1 prompt_assembler.py —— 四段式组装（文档第6节）

```
════ 系统 Prompt ════════
{L1 角色定义}
{L2 诊断锚点 + 置信度}
{L4 防越界指令}
════ 本轮 Prompt ════════
{L3 本轮动态指令}
{L3.plugin_guidance 插件嵌入指导}
════ 上下文 ════════════
{对话历史}
════ 当前 ══════════════
用户：{最新消息}
治疗师：
```

输出为 `list[Message]`。插件嵌入由 `plugin_guidance` 独立段落注入本轮 Prompt。

### 8.2 responder.py

- 调 `backend.chat(..., json_mode=True)`，模型返回 `{"reply": ..., "stage_complete": true/false, "reason": ...}`
- Pydantic 校验解析；失败则重试 1 次降级纯文本。
- `l4_substage` 由状态机给出。

### 8.3 summary_builder.py

构建 `L5Context` 与 `StructuredSummary`（含 evidence_refs）。

## 9. 回环A 编排与兜底（review）

### 9.1 reviewer.py —— 审核端口协议

`Reviewer.review(l5_context, review_history) -> ReviewResult`。真实 L5 与 `MockReviewer` 共用契约。

### 9.2 orchestrator.py —— 回环A

```
for attempt in 1..3:
    回复 = responder 生成（首轮原始 prompt；后续注入 fix_instruction）
    result = reviewer.review(...)
    全部 pass → 返回
    否则记录失败原因，追加 fix_instruction 重写
3 次仍 fail → 安全兜底回复（预设模板）
            + flags=["loop_A_fallback", "manual_review_required"]
            + 记录 3 次失败原因
```

- 重试只在本轮内进行，`conversation_history` 不写入被驳回草稿。
- 暴露 `attempt_count` 与失败原因追溯，支撑 V4-4（连续 3 轮未收敛 = 架构缺陷信号）。

### 9.3 接口自校验（V4-1 ~ V4-4 的 L4 侧）

`service.py` 返回前本地校验：`l4_substage` 枚举合法、`therapy_type` 与诊断一致（V4-3）、输出不含 L3 `forbidden` 词；失败即触发一次内部重写，实现"出口自净"。

## 10. 配置 / FastAPI 服务 / CLI 演示

### 10.1 config.py

Pydantic Settings 读取环境变量 + `.env`：
- `LLM_PROVIDER`（openai_compat/mock/local）
- `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
- `REVIEW_MODE`（mock/l5_endpoint）
- 重试次数、兜底模板

默认 `LLM_PROVIDER=mock` + `REVIEW_MODE=mock`，克隆即运行，无需密钥。

### 10.2 api/server.py（FastAPI）

- `POST /v1/l4/session`：创建会话（L1/L2/L4 + 可选 L3），返回 `session_id`
- `POST /v1/l4/turn`：一轮处理。请求体含 `session_id + L3 + 用户消息 + L1 SafetyPassport 安全标记`，响应体 `{reply, l5_context, summary, review_history}`
- `POST /v1/l4/rewrite`：外部回环A 入口（供 L5 编排场景备用，接收 fix_instruction）
- `GET /healthz`

### 10.3 cli/demo.py

交互式 CLI 演示：从 `examples/sample_diagnostic_report.json` 加载示例 L3 输入，走完 L4-L1→L4-L4 四阶段，显示每轮 `l4_substage`、`review_history`、`summary`。FakeLLM 跑通，无密钥。

## 11. 测试与验证

pytest 单测（全部用 MockBackend + MockReviewer，无网络）：

| 测试文件 | 覆盖点 |
|---|---|
| `test_schemas.py` | 契约字段校验、枚举约束、DiagnosticReport 示例解析 |
| `test_state_machine.py` | 默认推进、L3 覆盖、禁止回退、终态异常标记 |
| `test_prompt_assembler.py` | 四段式结构、插件嵌入段、fix_instruction 注入位置 |
| `test_review_loop.py` | 一次通过/重写后通过/3 次失败兜底 + manual_review_required |
| `test_service.py` | 端到端一轮处理：回复+L5Context+Summary，枚举合法、evidence_refs 非空 |
| `test_l5_validation.py` | V4-1 枚举、V4-3 therapy_type 一致性本地自校验 |

验证命令：

```
pip install -e .
pytest
python -m l4.cli.demo
uvicorn l4.api.server:app --port 8000
```

层内自查指标（CBT 干预步骤完整性 >90%，参照 Rollwage 96%）：`test_prompt_assembler.py` 断言 L3 指定的技术步骤完整出现在系统 Prompt 中，逐步累积覆盖率。

## 12. 设计要点与边界

- **L4 不做临床推理**：即使 L3 指令有误，L4 忠实执行不纠正，错误归因于诊断层（错误可溯源）。
- **七步结构是框架不是模板**：CBT 七步（议程→心情→连接→核心议题→作业→总结→反馈）作为结构骨架，L3 决定本轮焦点（核心议题内容），不做固定句式。
- **插件是嵌入不是拼接**：MBCT/ERP 等通过 plugin_guidance 自然融入对话。
- **可追溯**：evidence_refs 支撑从 L5 逆溯到 L1 的完整证据链。
- **模型无关**：LLMBackend 协议隔离模型实现，本地训练模型就绪后只加适配器。
