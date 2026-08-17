# Layer 4 治疗对话层（Treatment Dialogue Layer）

五层认知架构（安全评估 → MI评估 → 诊断推理 → **治疗对话** → 输出安全审核）的第四层。
本层接收 L3 诊断层下发的 L1-L4 治疗指令，将其"翻译"为自然、温暖、专业的治疗对话，
不做任何临床推理（临床推理与语言生成分离）。

## 模块结构

```
l4/
├── config.py         # 配置（环境变量前缀 L4_）
├── schemas/          # 层间数据契约（L1-L4 指令 / DiagnosticReport / L5Context / 摘要 / 审核）
├── llm/              # LLM 端口与适配器（OpenAI 兼容 / Mock / 本地模型占位）
├── session/          # L4-L1→L4-L4 子层状态机与会话历史
├── generation/       # prompt 四段式组装、回复生成、结构化摘要
├── review/           # 审核端口、Mock 审核器、回环A（≤3 轮重写+兜底）
├── service.py        # L4 门面：一轮完整处理
├── api/server.py     # FastAPI 服务
└── cli/demo.py       # CLI 演示
```

## 快速开始（无需 API 密钥）

```bash
pip install -e ".[dev]"
pytest                       # 跑全部单测（Mock 实现，无网络）
python -m l4.cli.demo --scripted   # 脚本化演示四阶段对话
python -m l4.cli.demo              # 交互式演示
uvicorn l4.api.server:app --port 8000   # 启动服务
```

## 接入真实 LLM（DeepSeek）

```bash
set L4_LLM_PROVIDER=openai_compat
set L4_LLM_BASE_URL=https://api.deepseek.com
set L4_LLM_API_KEY=sk-xxxx
set L4_LLM_MODEL=deepseek-chat
set L4_REVIEW_MODE=mock        # 真实 L5 就绪后替换为 l5_endpoint 或注入 reviewer
uvicorn l4.api.server:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/v1/l4/session` | 创建会话（role + DiagnosticReport + boundary + safety） |
| POST | `/v1/l4/turn` | 一轮处理（session_id + L3 指令 + 用户消息） |
| POST | `/v1/l4/rewrite` | 外部回环A 入口（携带 fix_instruction 重写） |

## 接口契约

- 输入：接口三 `DiagnosticReport`（L3→L4），见 `examples/sample_diagnostic_report.json`
- 输出：接口四 `ContextualSafetyContext`（L4+L3→L5）+ 结构化临床摘要（`StructuredSummary`，含 `evidence_refs` 证据链）
- 回环A：L5 fail → 注入 `fix_instruction` 重写，最多 3 轮，超限输出安全兜底回复并标记 `loop_A_fallback` + `manual_review_required`

## 接入本地训练模型

本地模型训练完成后，实现 `l4/llm/local.py`（vLLM/transformers），保持 `LLMBackend.chat()` 签名不变，
并设置 `L4_LLM_PROVIDER=local`，L4 其余代码零改动。

## 测试

```bash
pytest -v
```

单测全部使用 MockBackend + MockReviewer，无网络依赖。
