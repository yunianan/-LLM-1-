# DSM-5 失眠障碍补充

整理日期：2026-10-14。此目录是项目补充，原始 MentalKG 不变；不是 MentalKG 作者发布或专家审查的新增节点。

沿用原图谱的目录与字段格式：`disorder.json`、`diagnostic_criteria.json`、`symptom/`、`event.json`、`differential_diagnosis/`。保留活动标签 `Insomnia`，补充症状 ID S085–S087，不占用或覆盖原 S001–S084。

## 条件定位

| DSM-5 条件 | 本地 JSON 条目 |
|---|---|
| A：主诉及睡眠困难 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/insomnia_symptoms`；`additional_requirements/0` |
| A1–A3：具体表现 | `EN/symptom/10insomnia.json#/S085`、`#/S086`、`#/S087` |
| B：临床显著性 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/additional_requirements/1`；`EN/event.json#/functional_impairment_required` |
| C：频率 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/additional_requirements/2` |
| D：病程 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/min_duration` |
| E：睡眠机会 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/additional_requirements/3` |
| F–H：解释与排除条件 | `EN/diagnostic_criteria.json#/Insomnia/required_criteria/additional_requirements/4`、`/5`、`/6` |

运行时 `source_ref`、`criterion_source_refs` 和报告 `source_doc` 直接引用对应 JSON，不索引论文。

`functional_impairment_required=true` 在本补充中按独立 `event.json` 解释为显著痛苦或功能影响；不能据此要求二者同时存在。本补充不改变原 MentalKG 的 event 定义。

未达到病程条件时，不可将当前候选标为满足完整标准；短期问题及病程说明不在本候选中另建诊断节点。共存疾病本身不构成自动排除；鉴别文件只表达条件间的关系，不替代相关原节点的诊断标准。

## 建库依据

诊断条件为简明改写，依据 **DSM-5（2013）** 的公开派生资料；不混用 DSM-IV 或 DSM-5-TR。

- [SAMHSA，2016，Table 3.36，DSM-IV to DSM-5 Insomnia Disorder Comparison](https://www.ncbi.nlm.nih.gov/books/NBK519704/table/ch3.t36/)：仅使用 DSM-5 列核对 A–H 与儿童表现。本次完整表格通过网页检索结果读取；直接访问曾遇到浏览器检查。
- [APA，2013，Sleep-Wake Disorders](https://www.psychiatry.org/File%20Library/Psychiatrists/Practice/DSM/APA_DSM-5-Sleep-Wake-Disorders.pdf)：用于核对共存疾病与独立失眠诊断的关系，不作为完整标准表。

外部链接用于说明建库来源；系统取证定位使用上表的本地 JSON 条目。没有新增治疗规则，也没有把工程测试表述为专业医生审核。
