# 部署与复现

## 环境

```bash
conda activate HPO
python -m pip install -e '.[ml,dev]' --no-build-isolation
```

已验证环境：Python 3.10.19、PyTorch 2.6.0+cu124、Transformers 4.57.0、
FAISS 1.15.1、Sentence-Transformers 5.7.0。GPU 推理需在可访问 NVIDIA 设备的环境运行。

模型目录约定：

- `artifacts/models/biobert-base-cased-v1.2`
- `artifacts/models/SapBERT-from-PubMedBERT-fulltext`

## 验证实验

```bash
python scripts/make_split.py \
  --input PatientPheX-V1-A/PatientPheX-train.jsonl \
  --output configs/split_seed42.json --seed 42 --validation-size 16

CUDA_VISIBLE_DEVICES=0 python scripts/train_ner.py \
  --data PatientPheX-V1-A/PatientPheX-train.jsonl \
  --split configs/split_seed42.json \
  --model artifacts/models/biobert-base-cased-v1.2 \
  --output artifacts/ner/biobert_seed42 --epochs 5

CUDA_VISIBLE_DEVICES=0 python scripts/build_retrieval_index.py \
  --ontology PatientPheX-V1-A/hp.obo \
  --model artifacts/models/SapBERT-from-PubMedBERT-fulltext \
  --output artifacts/retrieval/sapbert_hpo
```

阈值评估和融合脚本分别为 `scripts/evaluate_ner.py`、
`scripts/evaluate_retrieval.py`、`scripts/evaluate_supervised_retrieval.py` 和
`scripts/run_dual_channel_experiment.py`。

## A 榜推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict_ner.py \
  --input PatientPheX-V1-A/PatientPheX-A.jsonl \
  --model artifacts/ner/biobert_full \
  --output artifacts/ner/biobert_full/a_spans.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/predict_submission.py \
  --input PatientPheX-V1-A/PatientPheX-A.jsonl \
  --training-data PatientPheX-V1-A/PatientPheX-train.jsonl \
  --ontology PatientPheX-V1-A/hp.obo \
  --ner-predictions artifacts/ner/biobert_full/a_spans.jsonl \
  --retrieval-model artifacts/models/SapBERT-from-PubMedBERT-fulltext \
  --retrieval-index artifacts/retrieval/sapbert_hpo \
  --output outputs/PatientPheX-A-task1-pred.jsonl
```

提交前运行 `hpo-pipeline validate`。该校验检查文档覆盖、PMID、必需字段、字符坐标、
重复实体、否定标记以及 HPO 分支。任务一不生成患者关联，`association` 保持空数组。

