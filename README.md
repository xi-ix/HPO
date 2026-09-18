# HPO
#数据集
\PatientPheX-V1-A
用于完成第一次测试提交

CHIP2026 任务一的无生成式 LLM 实现。系统由 HPO/训练词典、BioBERT 实体识别、
SapBERT + FAISS 标准化、否定规则和双通道仲裁组成。

## 当前结果

固定 seed=42 的 64/16 文档级划分：

| 实验 | Mention P/R/F1 | Document P/R/F1 |
|---|---|---|
| 词典基线 | 0.7713 / 0.5836 / 0.6644 | 0.8534 / 0.5874 / 0.6958 |
| 最终双通道 | 0.7707 / 0.6605 / 0.7114 | 0.8496 / 0.6919 / 0.7627 |

双通道 Mention F1 相对词典基线提升 7.07%。完整实验说明见
[`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md)。

## 快速检查

```bash
conda activate HPO
pip install -e . --no-build-isolation
pytest -q
hpo-pipeline validate \
  --input PatientPheX-V1-A/PatientPheX-A.jsonl \
  --prediction outputs/PatientPheX-A-task1-pred.jsonl \
  --ontology PatientPheX-V1-A/hp.obo
```

A 榜任务一提交文件为 `outputs/PatientPheX-A-task1-pred.jsonl`。运行流程、模型获取和
复现实验命令见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。
