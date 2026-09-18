# 下一轮改进方案

## 1. 结论与优先级

A 榜 Mention F1 为 0.69，距头部 0.07；Document F1 为 0.78，距头部仅 0.03。
这表明文档级概念覆盖已经接近头部，主要损失来自：实体边界不完全一致、同一 span 的 HPO
粒度选择错误、复合实体未完整处理。下一轮应优先优化 Mention，而不是无约束扩大词典召回。

本地 0.7114 到线上 0.69 的下降还说明当前 16 篇单划分对阈值选择偏乐观。下一轮第一项必须
是建立可靠的 OOF 评估，不应继续围绕同一验证集调参。

## 2. P0：五折 OOF 与错误账本

1. 按文档做 5 折交叉验证，每折 64/16，固定所有 fold ID。
2. 每折独立训练 NER，得到全部 80 篇的 out-of-fold 预测。
3. 统一统计严格 TP、边界重叠、精确边界 ID 错误、漏检、否定、复合和 `-1`。
4. 按 section、实体长度、是否词典命中、是否复合、HPO 深度分桶。
5. 阈值按五折 macro 平均选择，并报告均值、标准差和 bootstrap 置信区间。

验收：后续方案只在 OOF Mention F1 至少提升 0.01 且多数 fold 改善时进入 A 榜提交。

## 3. P1：边界模型升级

当前 BIO + token CE 容易把 `global developmental delays` 截成内部短实体。建议并行比较：

- BioBERT + CRF，显式学习合法 BIO 转移；
- span start/end 双头或 GlobalPointer，直接为候选跨度打分；
- PubMedBERT/BioLinkBERT-base，与 BioBERT 做 OOF 概率融合；
- 加入实体长度、左右边界和词典包含关系的辅助损失。

训练中对现有 98 类边界重叠错误做 hard-example oversampling，但仍只使用比赛训练标注。
推理时用模型跨度与词典跨度联合打分，不再用单一固定包含阈值。

预期：Mention F1 提升 0.02-0.04，Document F1 基本持平。

## 4. P1：上下文标准化重排器

SapBERT 在金 mention 上重排后 Top-1 仍为 0.8702，约 13% 存在标准化错误。构建判别式
cross-encoder，输入 `mention + 所在句 + 候选 HPO 名称/同义词/定义`，对 SapBERT Top-20
重排。正例来自比赛训练集，难负例来自同父节点、SapBERT 高相似候选和易混淆兄弟节点。

应加入层级目标：当父子候选都合理时，用上下文中的部位、程度、时间和具体修饰词选择更特异
术语。复合实体先判定是否需要多个 ID，再对每个成分标准化。

预期：Mention F1 提升 0.01-0.02，Document F1 提升约 0.01。

## 5. P2：复合与无 ID 实体

- 为训练集 69 个复合实体建立结构规则：并列词、逗号、`and/or`、斜杠和共享中心词。
- 增加 `single / compound / -1` 三分类头；只有 compound 时才输出分号 ID。
- 对 `-1` 只训练边界，不强制映射到相似 HPO，避免引入 Document FP。
- 统计 alt ID 和疑似标注异常，评估时同时保留官方原始口径与主 ID 规范化口径。

## 6. P2：集成与提交策略

最终用五折 NER 模型进行 span 投票或概率平均，全量训练模型只作为补充。根据 OOF 结果预先
冻结三种提交：balanced、high-precision、high-recall。A 榜每日有限提交次数只用于验证
离线已经确定的三个 operating point，不再用排行榜逐点搜索阈值。

优先目标：先把 OOF Mention F1 稳定提升到 0.73 以上，再提交 A 榜；若 A 榜达到 0.73-0.74，
继续做 cross-encoder 与模型集成，目标接近 0.76/0.81。

