# Lab 1：BioBERT + SapBERT 双通道基线

本目录冻结 CHIP2026 任务一第一轮实验的代码、配置、模型产物、验证结果和 A 榜提交。
后续实验不应直接修改本目录；新方案应在新的实验目录中开发，并以本目录为对照。

## 成绩

| 数据 | Mention F1 | Document F1 | 说明 |
|---|---:|---:|---|
| 本地固定验证集 | 0.7114 | 0.7627 | 训练集按文档 64/16，seed=42 |
| A 榜线上 | 0.6900 | 0.7800 | 2026-09-18 用户提交反馈 |
| A 榜头部 | 0.7600 | 0.8100 | 同期用户提供的榜首参考值 |

线上与头部的绝对差距为 Mention `0.07`、Document `0.03`。本地到线上 Mention 下降
`0.0214`，说明单次 16 文档验证存在方差或阈值过拟合；Document 反而上升 `0.0173`，
系统概念覆盖尚可，首要瓶颈是精确实体边界和 mention 级标准化。

## 内容

- `src/`：核心库代码。
- `scripts/`：训练、索引、评估、融合和预测脚本。
- `configs/split_seed42.json`：固定验证划分。
- `reports/`：实验报告和机器可读网格结果。
- `outputs/PatientPheX-A-task1-pred.jsonl`：实际提交的 A 榜预测。
- `artifacts/ner/`：验证模型和全部训练数据最终模型。
- `artifacts/retrieval/sapbert_hpo/`：FAISS 索引和映射表。
- `ONLINE_RESULTS.json`：线上成绩记录。
- `ARTIFACT_CHECKSUMS.sha256`：关键产物哈希。
- `NEXT_STEPS.md`：下一轮改进方案。

BioBERT/SapBERT 原始预训练仓库未重复放入本目录，仍位于项目根目录
`artifacts/models/`。PatientPheX 原始数据也只保留一份，路径为根目录 `PatientPheX-V1-A/`。
本目录中的大权重和索引是独立物理副本，可与根目录后续实验分别修改或删除。
