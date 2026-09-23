# 心理诊疗 LLM — 五层认知架构

五层认知架构（临床推理与语言生成分离）的统一实现仓库：

```
L1 安全评估 → L2 MI评估 → L3 诊断推理(CLINICR) → L4 治疗对话 → L5 输出安全审核
```

## 仓库结构

```
├── layers/                  # 各层实现（L2 待接入）
│   ├── layer1/              # L1 安全评估层（关键词+LLM语义+越狱检测 → 风险融合）
│   ├── layer3/              # L3 诊断推理层（CLINICR三步 + 疗法选择 + L1-L4指令生成）
│   ├── layer4/              # L4 治疗对话层（状态机/回环A/prompt组装/FastAPI）
│   └── layer5/              # L5 输出安全审核层（上下文规则生成 + 一次调用三检）
├── shared/                  # 公共模块
│   ├── llm_client.py        # DeepSeek 客户端（DEEPSEEK_API_KEY 环境变量）
│   ├── prompts/             # 共享 prompt 模板
│   └── schemas/             # 层间契约 TypedDict（SafetyPassport 等）
├── examples/                # 演示脚本 demo_l1.py / demo_l3.py / demo_l5.py
├── tools/
│   └── verify_interfaces.py # 层间接口契约验证（35 项检查，含端到端）
├── tests/                   # L4 单元测试（54 个）
└── docs/                    # 设计文档与实现计划
```

## 层间接口（严格按架构文档）

| 接口 | 格式 | 位置 |
|---|---|---|
| 接口一 L1→L2 | SafetyPassport（13 字段） | `layers/layer1/risk_fusion.py` |
| 接口二 L2→L3 | ClinicalIntake | 待 L2 接入 |
| 接口三 L3→L4 | DiagnosticReport（结构化诊断+疗法+证据链） | `layers/layer3/api.py` |
| 接口四 L4+L3→L5 | ContextualSafetyContext → L5 审核输出 | `layers/layer5/reviewer.py`、`layers/layer4/schemas/l5_context.py` |

L4 的 Pydantic schema（`layers/layer4/schemas/`）是各接口的权威契约；`layers/layer4/review/l5_adapter.py` 将真实 L5 包装为 Reviewer 协议，回环A（≤3 轮重写+兜底）在 L4 内编排。

## 快速开始（无需 API 密钥）

```bash
pip install -e ".[dev]"
pytest                                    # L4 单测 54 个
python tools/verify_interfaces.py         # 三层接口契约 + 端到端 35 项
python examples/demo_l3.py                # L3 诊断演示（无网络）
```

## 接入真实 DeepSeek

```bash
set DEEPSEEK_API_KEY=sk-xxxx
python examples/demo_l1.py "我最近压力很大"          # L1 演示（调真实 API）
python examples/demo_l5.py                            # L1+L5 演示
```

L4 服务：`uvicorn layers.layer4.api.server:app --port 8000`（配 `L4_LLM_PROVIDER=openai_compat` 等环境变量）。

## 已实现的关键机制

- 回环A（L5 打回 → L4 重写 ≤3 轮 → 安全兜底 + `manual_review_required`）
- L4 子层状态机 L4-L1→L4-L4（L3 指令可覆盖，回退打标记）
- V4 本地自校验（禁止词/绝对禁令/疗法一致性）
- 证据链 evidence_refs（支撑端到端逆溯检验）
