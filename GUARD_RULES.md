# Patient Agent 事实守卫

范围：模拟患者输出校验，不是实际患者的安全评估，不更改 L1–L5。

流程：生成候选 → 硬规则 → 全文事实复核 → 通过才显示/发送；失败按原事实有限重说，仍失败就停止。开场也执行相同校验。

硬规则检查：事实编号存在、披露权限、否认标记、禁止内容、风险升级、数字与单位、病程、严重度；量表必须有当前题目，题号/范围/病例答案均一致。不能借用其他症状或量表中的数字证明本轮病程。

真实 API 模式：复用已配置模型，额外调用一次独立的事实复核请求。每段返回 quote（原文）、status（supported 有依据 / neutral 无事实过渡 / unsupported 未记载 / contradicted 矛盾）、reference_ids（正式病例依据编号）、reason（简短理由）。程序检查全文覆盖、引用有效性；超时、非法 JSON 或遗漏覆盖均不放行。医生的问题不能作为患者经历的依据。

离线模式：只接受可以完整核对的病例参考台词，允许标点和中文数字形式差异；不把文字相似度当作事实正确。

GuardResult：ok 是否放行；violations 违规列表。每项 code 违规类别、message 原因、evidence 命中内容、blocking 是否阻断、retryable 是否允许重说。复核不可用不继续重说。

TurnRecord：submitted 是否已确认提交（旧记录默认 true）；rejected_text 被拒绝候选文本；final_text 最终通过的候选；attempts 全部生成/复核尝试；response_payload 后端结构化返回。阻断或提交失败时 patient_text 为空。守卫停止是测试停止，不伪造临床疗程结束。

## 完整证据链

本轮已有事实编号 → 正式病例依据快照 → 分段候选 → 每段复核 → 全部重试记录 → 提交状态及返回。

### 编号与依据

病例 JSON 不修改。复用 `symptom:<symptom_id>`、`background:<fact_id>` 等现有派生编号；病程字段派生为 `symptom:<symptom_id>:duration`。`source_values` 的键是原始字段路径，值是当时的原始数据。数组路径只负责定位，症状/背景事实的身份由已有 ID 决定。开场、变化轨迹、信念等原设计使用数组索引的引用，只在记录中的病例版本内解释；不宣称跨版本稳定。

`FactEvidence` 字段：fact_id 依据编号；scenario_id 病例编号；scenario_version 病例版本；value 本轮可使用的参考内容；status 为 present 存在、denied 明确否认、non_factual 过渡话术；source_values 原始来源快照。`policy.neutral.*` 是程序内置的非事实话术来源，不伪装成病例临床字段。未记录信息不会被自动生成成否认事实。

### 生成与复核

API 模式生成 JSON 的 `segments`：segment_id 片段编号、text 原文、kind 为 factual/non_factual、fact_ids 为本轮依据编号列表。程序拼接 text 为唯一候选文本。无依据的非事实片段必须引用空列表，但仍接受语义复核。生成 schema 不合法、片段不完整、引用未知/非授权编号、生成依据快照被改动均阻断。

复核结果保存在 `GuardResult.semantic_review.claims`，除 quote/status/reference_ids/reason 外还保存 segment_id。必须逐段保持编号和完整原文，通过的 reference_ids 必须与生成绑定一致。GuardResult 还保存 authorized_evidence、negative_evidence（仅作矛盾检测、不授权披露）、review_mode、semantic_request（实际复核提示词）、semantic_prompt_version、semantic_raw_output。离线模式的 review_mode 为 exact_reference，只作确定性台词核对，不假装调用了 LLM。

### 每次尝试与提交

`GenerationAttempt`：attempt_id 当前尝试编号；previous_attempt_id 上一次尝试；attempt_number 次序；doctor_question 医生问题；selection_reason/selected_fact_id 选择原因与事实；authorized_evidence 正式核对依据；generation_evidence 生成时使用的快照；observed_events 当时披露条件；candidate_text/segments/raw_output 候选及原始生成返回；generation_error 安全错误类型；generated_at/reviewed_at 时间；generation_elapsed_ms/review_elapsed_ms 耗时；model_id 配置模型 ID；prompt_version/prompt_system/prompt_user 生成提示词版本及实际内容；action_payload 非文本动作参数；guard_result 完整复核结果；submission_status/submitted_at 提交状态及时间。失败候选保留但不得加入有效对话历史。

submission_status：not_submitted 尚未提交；submitted 已收到正常后端返回；blocked 守卫阻断；request_failed_unknown 后端请求异常，无法确认是否收到；displayed_only 手动医生模式只在终端展示、未提交后端。请求失败不等于病例事实错误，也不自动重复提交。

自动回放的记录仍在 `runs/.../recorder.json`；手动测试新增 `runs/manual-.../recorder.json`，每轮更新，启动时打印位置。均可通过 SessionRecorder.dump/load 导出恢复记录；这不等于恢复后端会话。记录含病例、提问和提示词，仅作开发调试，不记录 API 密钥或内部思维链。

局限：语义复核仍依赖模型，不能保证零幻觉；同一模型的独立请求不是独立训练的模型。额外复核增加费用和延迟。关键词事实选择器本次不修改，未选到事实时仍可能拒答或被守卫阻断，而不是自动补全经历。
