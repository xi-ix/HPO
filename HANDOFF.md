# CHIP2026 PatientPheX 项目交接

## 1. 当前状态

已完成任务一的首轮可运行系统（Lab1），包括词典候选、BioBERT 实体识别、SapBERT +
FAISS 标准化、否定规则、双通道仲裁、严格提交校验和 A 榜预测。

当前还没有训练任务二模型，也没有训练计划中的两个 LoRA Adapter。后续建议在共享基础模型上
分别训练 Task1 收尾 Adapter 和 Task2 患者关联 Adapter。

## 2. 已确认成绩

| 数据 | Mention F1 | Document F1 |
|---|---:|---:|
| 本地固定 64/16 验证集 | 0.7114 | 0.7627 |
| A 榜线上 | 0.6900 | 0.7800 |
| 同期榜首参考 | 0.7600 | 0.8100 |

本地验证划分固定在 `configs/split_seed42.json`。A 榜成绩由实际提交
`lab1/outputs/PatientPheX-A-task1-pred.jsonl` 获得。

## 3. 仓库入口

- `src/hpo_pipeline/`：核心 Python 包。
- `scripts/`：训练、评估、索引、融合和提交生成脚本。
- `tests/`：基础单元测试。
- `docs/DEPLOYMENT.md`：环境及完整命令。
- `docs/API.md`：命令行和 Python 接口。
- `reports/EXPERIMENT_REPORT.md`：Lab1 方法、参数、消融和本地成绩。
- `lab1/`：第一轮实验冻结快照与下一步方案。
- `lab1/任务一计划书.md`、`CHIP2026_任务一任务书.md`：方案与正式任务要求。

## 4. 模型交付

模型没有放进普通 Git。打包文件位于本机：

```text
release/HPO_lab1_models_20260918.tar.zst
release/HPO_lab1_models_20260918.tar.zst.sha256
```

整包 SHA-256：

```text
9fe07559681f4ac3df3b1cfa09e7a20ab545880887a57346333a16be0cad2b98
```

压缩包约 887MB，适合作为 GitHub Release Asset 上传，不适合普通 Git commit。包内包含：

- 全部 80 篇训练文档微调的 `biobert_full`；
- 干净的 SapBERT 权重和 tokenizer；
- HPO FAISS 索引、surface 映射和元数据；
- 包内逐文件 `SHA256SUMS` 与说明。

解压和校验：

```bash
tar --zstd -xf HPO_lab1_models_20260918.tar.zst
cd HPO_lab1_models_20260918
sha256sum -c SHA256SUMS
```

## 5. 环境与验证

已验证环境为 Python 3.10.19、PyTorch 2.6.0+cu124、Transformers 4.57.0、
FAISS 1.15.1。完整版本见 `lab1/environment.txt`。

```bash
conda activate HPO
python -m pip install -e . --no-build-isolation
pytest -q
ruff check src scripts tests

hpo-pipeline validate \
  --input PatientPheX-V1-A/PatientPheX-A.jsonl \
  --prediction lab1/outputs/PatientPheX-A-task1-pred.jsonl \
  --ontology PatientPheX-V1-A/hp.obo
```

原始数据未修改，哈希在 `data_checksums.sha256`。最终 A 榜文件包含 20 篇文档、1,392 个
实体，坐标、字段、文档覆盖和 HPO 分支校验均通过。

## 6. 当前方法摘要

1. HPO/训练 mention 词典进行最长匹配和保守单复数扩展。
2. BioBERT 使用 BIO 标签、384 token 滑窗、96 token 重叠，训练 4 轮。
3. SapBERT 编码 43,582 个 HPO 名称/同义词并构建 FAISS 内积索引。
4. 用训练 mention、词元重叠和 HPO 深度重排候选。
5. 高置信 NER span 严格包含词典短 span 时允许边界修正。
6. NegEx 风格规则处理否定；最终 ID 必须通过 HPO 目标分支白名单。

## 7. 已知问题

- A 榜 Mention F1 比本地低 0.0214，单一 16 篇验证集存在方差和阈值过拟合风险。
- Document F1 距榜首仅 0.03，主要瓶颈是精确 mention 边界而非文档级概念覆盖。
- 复合实体只有 69 个训练样本，当前处理不足。
- SapBERT 金 mention 重排后 Top-1 为 0.8702，仍有 HPO 粒度与近义概念混淆。
- 否定 accuracy 为 0.9644，但 precision 偏低，复杂否定范围仍需模型化。
- 训练标注含少量旧 alt ID 和疑似异常 ID，代码已统一执行目标分支规范化。

## 8. 下一步建议

优先级详见 `lab1/NEXT_STEPS.md`：

1. 先建立 5 折 OOF，冻结所有 fold 和阈值，减少线上回落。
2. 比较 BioBERT+CRF、GlobalPointer 或 span start/end 模型，重点解决完整边界。
3. 训练上下文 cross-encoder 重排 SapBERT Top-20。
4. 单独建模 single/compound/no-ID。
5. 再引入一个 <=10B 基础模型和两个独立 LoRA：Task1 候选收尾、Task2 患者关联。
6. Task2 信号只能辅助 Task1 仲裁，不能因为实体未关联患者就从全文任务一中删除。
