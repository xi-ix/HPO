# CHIP2026 任务一实验报告

## 1. 实验设置

- 数据：PatientPheX 训练集 80 篇，未使用外部人工标注数据。
- 划分：seed=42，按文档随机划分 64 篇训练、16 篇验证；ID 固化于
  `configs/split_seed42.json`，同一文档不会跨集合。
- 本体：随数据提供的 HPO v2026-06-23，仅允许 `HP:0000118` 分支。
- 模型：BioBERT `dmis-lab/biobert-base-cased-v1.2` 与 SapBERT
  `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`，约 0.22B 参数。
- 评估：普通实体必须字符边界和 HPO ID 同时完全一致；否定实体不作为正例；复合 ID
  按分号拆分。Document 指标按文档 HPO 集合计算。

## 2. 方法

1. 解析 HPO OBO，提取名称、同义词、父子层级和替代 ID，只保留目标分支。
2. 符号通道使用 HPO 名称/同义词与训练 mention 最长匹配，并生成保守的末词单复数变体。
3. BioBERT 使用 BIO 标签、384 token 滑窗、96 token 重叠训练；batch=16，lr=3e-5，
   AdamW，类别权重指数 0.5，固定 seed=42。最佳验证轮为第 4 轮。
4. SapBERT 对 43,582 条 HPO 名称/同义词编码，CLS 池化和 L2 归一化后建立
   FAISS `IndexFlatIP`；额外索引训练集 mention，但先规范到当前 HPO 主 ID。
5. 候选重排融合 SapBERT 相似度、词元重叠和 HPO 深度。双通道以置信度阈值过滤，
   仅在高置信 NER span 严格包含词典短 span 时进行边界替换。
6. NegEx 风格规则处理否定，作用域在当前分句内。表格标题/说明不作为正文候选。

最终阈值：NER 0.75、SapBERT 原始余弦 0.90、映射一致边界 0.95、严格包含边界 0.95。

## 3. 结果

| 实验 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 原始词典基线 Mention | 0.7797 | 0.5594 | 0.6514 |
| 单复数扩展词典 Mention | 0.7713 | 0.5836 | 0.6644 |
| BioBERT 纯边界检测 | 0.6761 | 0.6929 | 0.6844 |
| 最终双通道 Mention | 0.7707 | 0.6605 | 0.7114 |
| 最终双通道 Document | 0.8496 | 0.6919 | 0.7627 |

最终 Mention 计数：TP=790、FP=235、FN=406。双通道相对单复数词典基线的 Mention
F1 提升 7.07%，超过计划书要求的 3%。输出 ID 全部通过目标 HPO 分支校验，幻觉率 0%。

SapBERT 金标准 mention 检索：Recall@1=0.8437、Recall@5=0.9436、Recall@10=0.9616。
加入训练 mention、词元重叠及层级重排后 Top-1=0.8702。否定规则在固定验证集上的
accuracy=0.9644、precision=0.4179、recall=0.8750。

## 4. 消融与结论

- 类别权重指数 0.0/0.25/0.35 的校准后 NER F1 分别为 0.6866/0.6831/0.6862；
  但接入系统后均未超过指数 0.5，说明互补性比单模块 F1 更重要。
- 允许任意 NER/词典重叠替换会降低 F1；限制为严格包含关系后 F1 明显提升。
- SapBERT 层级/词元重排提高金 mention Top-1，但最终收益受实体边界误差约束。
- Mention precision 和 F1 达到计划目标（>=0.75、>=0.71）；recall=0.6605，未达到
  原计划 0.68。后续优先改进复合表型和上下文化长边界，而不是降低检索阈值。
- 数据没有家族史标签，且正式任务要求全文表型而非患者归属，因此未报告家族史过滤
  accuracy，也未过滤家族史 mention。

## 5. 验证集预测文件

以下文件均对应固定 `seed=42` 的 16 篇文档验证集，文档 ID 由
`configs/split_seed42.json` 中的 `validation_pmc_ids` 定义。每行是一篇文档，包含预测的
`entities` 和空的 `association`；它们是评估输入，不是新的训练数据。

| 文件 | 含义 | 验证集实体数 |
|---|---|---:|
| `outputs/validation_lexicon.jsonl` | 原始 HPO/训练 mention 词典基线 | 909 |
| `outputs/validation_lexicon_inflected.jsonl` | 词典基线加保守单复数扩展 | 957 |
| `outputs/validation_dual_channel.jsonl` | 最终 BioBERT + SapBERT 双通道结果 | 1,086 |
| `outputs/validation_dual_weight000.jsonl` | NER 不使用类别权重的消融 | 1,062 |
| `outputs/validation_dual_weight025.jsonl` | NER 类别权重指数 0.25 的消融 | 1,059 |
| `outputs/validation_dual_weight035.jsonl` | NER 类别权重指数 0.35 的消融 | 1,058 |

这些文件与同名 `reports/generated/*.json` 指标报告配套使用。验证集金标准仍来自原始
`PatientPheX-train.jsonl`，通过固定划分读取；预测文件本身不复制或修改金标准。

详细机器可读结果位于 `reports/generated/`。A 榜没有公开答案，不能在本地计算分数；
`outputs/PatientPheX-A-task1-pred.jsonl` 仅报告格式校验通过，线上成绩需提交平台获得。
