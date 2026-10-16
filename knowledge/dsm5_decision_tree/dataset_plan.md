# DSM-5 决策树候选生成：病例集方案

日期：2026-10-14。阶段：来源和处理工具已确定；正式专家病例导入数为 0。

## 选择依据

主要病例来源选 **John W. Barnhill 主编的 DSM-5 Clinical Cases（2014 原版）**。其与项目的 DSM-5 标准一致，包含案例叙述和作者诊断/讨论。APA 期刊书评记载该书有 103 个案例：[书评](https://psychiatryonline.org/doi/10.1176/appi.ajp.2014.14020214)、[原书入口](https://psychiatryonline.org/doi/book/10.1176/appi.books.9781585624836)。

[方法论文](https://doi.org/10.1016/j.psychres.2025.116844)使用的是 TR 版病例集，106 例筛到 93 例，按主要 DSM 类别分层为 38/55。本项目复用其**作者标签、按覆盖范围筛选、按主要诊断类别分层、开发与测试隔离**的组织方法，不复制其样本数。开发集用于节点提示词调整，不意味着微调模型权重。

本轮未取得并导入可用的原版完整病例文本，也没有生成任何冒充专家病例的数据。源书取得后还需逐例整理画像、保留作者诊断及页码、确认标签映射。测试夹具是工程数据，不能计入正式样本数。

## 纳入与标签

1. 保留源书案例 ID、版本、页码、作者主要诊断原文及所有有把握映射的候选标签。
2. 主诊断须属于当前 24 标签，且画像有可追溯症状/病程/功能信息；主要诊断在范围外者记录排除理由。论文排除的四章不能直接代替本系统覆盖筛选。
3. ADHD 只写泛称而没有表现型时，不自动补成 D001；分裂情感障碍缺亚型时也不猜测。D008/D010、D013/D015 是层级相关标签，映射应由人工预先确定并保留理由，禁止为提高指标临时改标签。
4. 输入只包含病例叙述中可支持的事实。去掉诊断标题、作者诊断、诊断讨论和答案线索；未叙述的信息保持未知。画像需经人工核对，模型抽取结果不能自动成为金标准。
5. 主要诊断的 DSM 分类用于分层。这里涉及九个 DSM 类别；UI 也已拆分为九个展示组，但科研分层仍使用 DSM 类别定义，睡眠与焦虑分别处理。
6. 同一案例的多个问答、改写或不同版本必须合并到一个来源案例单位后划分，避免训练/测试交叉。

## 导入 JSONL

每行一个经整理的案例，以下仅是字段模板，不是病例：

```json
{
  "case_id": "来源书案例ID",
  "source_edition": "DSM-5",
  "source_ref": "原书版次、章节及页码",
  "author_primary_diagnosis": "作者诊断原文",
  "primary_label": "经核对的24标签之一",
  "gold_labels": ["经核对的标签"],
  "primary_dsm_category": "下列类别之一",
  "mapping_reviewed": true,
  "input_reviewed_no_diagnosis_leak": true,
  "vignette": "仅病例叙述，去掉诊断和讨论",
  "profile": {"按L2至L3接口整理的字段": "附真实来源证据节点"}
}
```

类别：`neurodevelopmental`、`schizophrenia_spectrum`、`bipolar`、`anxiety`、`depressive`、`obsessive_compulsive_related`、`trauma_stressor_related`、`feeding_eating`、`sleep_wake`。

```powershell
.\.venv\Scripts\python.exe tools/prepare_dsm5_case_split.py <人工整理的cases.jsonl> <输出目录> --seed 20261014
```

默认开发比例取论文的 `38/93`，各类别取整并尽可能同时进入两组。少于两例的类别只能进入开发组，并在报告明确列出；不能称该类别已被测试。脚本输出 dev/test 的 `.inputs.jsonl`、`.gold.jsonl` 和 `split_report.json`，输入文件只包含案例 ID 与画像，金标准独立保存。重复病例、未核对映射、TR 版、类别错配均会拒绝或列出原因。

正式病例不应放入候选运行时知识库或提示词中。若以后用专家案例 few-shot，只能从开发集选，并固定排除对应测试案例。

## 评测边界

候选层单独评估：24 标签覆盖率、候选集合 PPV、Top-K 召回、Top-K 命中率、空候选比例、每个标签召回，以及信息缺失与共病病例表现。候选层的 PPV/召回与经过奖励筛选的最终诊断指标分别报告。

`tools/evaluate_candidate_predictions.py` 接收独立金标准和候选输出 JSONL，不调用奖励模型。每条预测应为 `{"case_id":"...","candidates":[{"disorder":"D013","confidence":0.7}]}`，候选按支持度稳定排序。输出是相对于已标注标签的严格节点匹配结果；未标注的共病不自动等于临床上不存在，应人工复核错误候选。

```powershell
.\.venv\Scripts\python.exe tools/evaluate_candidate_predictions.py <test.gold.jsonl> <predictions.jsonl> --top-k 5
```

固定模型版本、提示词/树版本、抽样种子和标签映射后再跑留出测试集；不能边看测试结果边改节点后继续把它称为独立测试。论文的性能提升数字不作为本系统已经达到的结果。
