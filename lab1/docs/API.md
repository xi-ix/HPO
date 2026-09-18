# 接口说明

## 命令行

安装后入口为 `hpo-pipeline`：

- `inspect-ontology --ontology PATH`：统计 HPO 本体和目标分支。
- `predict --input PATH --ontology PATH --training-data PATH --output PATH`：运行轻量词典基线。
- `validate --input PATH --prediction PATH --ontology PATH`：严格校验提交文件。
- `evaluate --gold PATH --prediction PATH`：计算 Mention/Document P、R、F1。

训练与双通道推理使用 `scripts/` 中的显式实验脚本，参数和产物路径可由 `--help` 查询。

## Python

```python
from hpo_pipeline.lexicon import LexiconMatcher
from hpo_pipeline.ontology import HPOOntology
from hpo_pipeline.pipeline import PhenotypePipeline

ontology = HPOOntology.from_obo("PatientPheX-V1-A/hp.obo")
matcher = LexiconMatcher(ontology, training_documents)
pipeline = PhenotypePipeline(matcher)
prediction, trace = pipeline.predict_document(document)
```

`prediction` 与赛事任务一格式一致，包含 `pmc_id`、`pmid`、`entities` 和空的
`association`。`trace` 额外包含置信度、来源、否定状态和候选 ID，仅用于审计，不写入提交。

密集检索接口：

```python
from hpo_pipeline.retrieval import DenseRetriever

retriever = DenseRetriever(model_path, index_directory, device="cuda")
candidates = retriever.search(["developmental delay"], top_k=10)
```

每个候选包含 `identifier`、命中的 HPO `surface` 和余弦 `score`。最终输出前必须调用
`validate_submission` 或命令行 `validate`，以保证字符坐标和 HPO 分支合法。

