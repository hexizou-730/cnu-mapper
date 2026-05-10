# CNU Mapper 简短报告

## 1. 项目目标

本项目的目标是将学术文本描述分类到法国 CNU 学科部门
(`Conseil National des Universites`)。输入可以是一段论文摘要或课程描述，
输出是一个或多个 CNU section code。

目前主要实现的是一个先预测 DDC、再映射到 CNU 的流程：

```text
文本描述 -> DDC 预测 -> DDC-to-CNU 映射 -> CNU sections
```

这样设计的原因是原始论文数据中包含 Dewey Decimal Classification
(DDC) 元数据。因此，真正可以直接监督学习和评估的标签是 DDC。CNU 并不是
原始数据中由专家人工标注好的标签，而是通过 `dewey_to_cnu.py` 中的规则映射
从 DDC 推导出来的。

## 2. 数据集

实验使用本地生成的 `theses_labeled.csv`。该文件由法国官方论文数据生成，
因为体积较大，所以没有提交到 GitHub。

数据预处理逻辑在 `label_theses.py` 中实现。主要步骤包括：

- 保留长度在 100 到 4000 字符之间的法语摘要。
- 使用摘要前 200 个字符去除近似重复数据。
- 从原始元数据中提取 DDC code。
- 使用 `dewey_to_cnu.py` 将 DDC code 映射到 CNU section。

因此，这个数据集对 CNU 来说是弱标签数据：DDC 标签来自原始元数据，CNU 标签
来自人工整理的 DDC-to-CNU 映射表，而不是专家直接标注。

## 3. 数据统计

| 项目 | 数值 |
|---|---:|
| 样本数 | 331,608 |
| 不同 DDC code 数量 | 92 |
| 官方 CNU section 数量 | 80 |
| 当前数据中出现的 CNU section 数量 | 45 |
| 每条样本平均 DDC 标签数 | 1.000 |
| 每条样本平均 CNU 标签数 | 1.240 |
| 摘要长度中位数 | 1,655 字符 |
| 摘要长度平均值 | 1,706 字符 |

在当前生成的数据集中，DDC 是单标签任务。最终的 CNU 输出是多标签任务，因为
一个 DDC code 可能映射到一个或两个 CNU sections。

数据分布明显不均衡。样本最多的 DDC code 集中在工程、物理、生物、化学、法学、
医学、计算机科学和心理学等方向。

出现最多的 DDC code：

| DDC code | 样本数 | 占比 |
|---:|---:|---:|
| 620 | 40,541 | 12.23% |
| 530 | 37,313 | 11.25% |
| 570 | 33,202 | 10.01% |
| 540 | 23,215 | 7.00% |
| 340 | 19,229 | 5.80% |
| 610 | 18,378 | 5.54% |
| 004 | 17,582 | 5.30% |
| 150 | 15,239 | 4.60% |

出现最多的映射后 CNU section：

| CNU code | Section | 样本数 | 占比 |
|---:|---|---:|---:|
| 62 | Energy engineering and process engineering | 45,882 | 13.84% |
| 28 | Dense media and materials | 38,567 | 11.63% |
| 64 | Biochemistry and molecular biology | 33,202 | 10.01% |
| 65 | Cell biology | 33,202 | 10.01% |
| 32 | Organic, inorganic and industrial chemistry | 23,215 | 7.00% |
| 02 | Public law | 19,327 | 5.83% |
| 01 | Private law and criminal sciences | 19,229 | 5.80% |
| 27 | Computer science | 18,696 | 5.64% |

当前数据只覆盖了 80 个官方 CNU sections 中的 45 个。剩下的 35 个官方 CNU
sections 在当前弱标签数据里没有样本，因此监督模型无法学习这些从未出现过的
类别。

更详细的数据统计见 `DATASET_STATISTICS.md`。

## 4. 已实现的方法

### 4.1 Random Baseline

Random baseline 会从训练集中出现过的 DDC code 中随机选择一个作为预测结果。
它的作用是作为最低参考线，用来说明完全不学习文本和标签关系时的效果。

### 4.2 Majority Baseline

Majority baseline 总是预测训练集中出现次数最多的 DDC code。由于数据集类别
分布不均衡，这个简单方法也会得到非零分数，因此它是一个有意义的基础对比。

### 4.3 DDC + Logistic Regression

第一个可训练模型使用 TF-IDF 特征和 One-vs-Rest Logistic Regression。它为
每个 DDC code 训练一个二分类器，然后把预测出的 DDC code 映射到 CNU section。

主要设置：

- TF-IDF word unigram 和 bigram
- 最大词表规模：100,000
- `min_df=3`
- Logistic Regression 参数 `C=2.0`
- 使用 class-balanced 二分类器
- 在验证集上自动校准全局阈值

### 4.4 DDC + Multi-Layer Perceptron

第二个可训练模型使用 TF-IDF 特征和 scikit-learn 的 `MLPClassifier`。与
Logistic Regression 不同，MLP 使用一个神经网络直接预测完整的 DDC indicator
vector。

主要设置：

- TF-IDF word unigram
- 最大词表规模：20,000
- 一个 64 hidden units 的隐藏层
- Adam optimizer
- 在验证集上自动校准全局阈值

### 4.5 DDC + XLM-RoBERTa Transformer

项目中也新增了基于 PyTorch/Transformers 的 `xlm-roberta-base` 路线。这个模型
仍然先预测 DDC code，再使用同一套 `dewey_to_cnu.py` 映射到 CNU。

该方法适合后续深度学习实验。目前本报告中最稳定的完整评估表主要展示
Random、Majority、Logistic Regression 和 MLP。

### 4.6 LLM Classifier

项目还包含一个基于 OpenRouter 的 LLM classifier。这个方法不训练本地模型，
而是把官方 CNU 列表放入 prompt，让大模型直接返回一到三个 CNU code。

LLM 方法目前更适合定性测试和人工案例分析。由于全量评估需要大量外部 API
调用，所以没有放入当前的完整数值评估表。

## 5. 评估方案

数值评估使用 80/20 train-test split。对于可训练模型，训练部分会进一步划分为
fit split 和 validation split。validation split 用于校准一个全局 DDC 概率阈值。

完整评估的数据划分：

- Baseline train split：265,286 条样本
- 可训练模型 fit split：212,228 条样本
- 阈值调优 validation split：53,058 条样本
- Test split：66,322 条样本

报告中的指标包括：

- **DDC Top-1 Accuracy**：最高分的 DDC 预测是否正确。
- **DDC Top-3 Accuracy**：真实 DDC 是否出现在分数最高的前三个 DDC 候选中。
- **Mapped CNU Micro F1**：把 DDC 预测结果映射到 CNU 后得到的下游代理指标。

DDC 指标是主要监督评估指标，因为 DDC code 真实存在于源数据元数据中。
Mapped CNU Micro F1 可以帮助理解最终 CNU 输出的效果，但它不是基于专家人工
标注 CNU 的直接评估。

## 6. 实验结果

| Model | DDC Top-1 Acc | DDC Top-3 Acc | Mapped CNU Micro F1 |
|---|---:|---:|---:|
| Random | 1.12% | 1.12% | 0.0257 |
| Majority | 12.29% | 12.29% | 0.1247 |
| LogReg | 62.72% | 89.13% | 0.6650 |
| MLP | 52.21% | 77.27% | 0.5663 |

Logistic Regression 是当前完整评估中表现最好的方法。它明显优于 Random 和
Majority baseline，说明模型确实从摘要文本中学习到了和学科相关的有效信号。

MLP 也明显优于两个简单 baseline，但低于 Logistic Regression。一个可能原因是
稀疏 TF-IDF 文本特征非常适合线性分类器；MLP 虽然更复杂，但不一定能从这种
高维稀疏表示中获得额外收益。

## 7. 讨论与限制

这些结果需要结合数据构建方式来理解。DDC 标签来自元数据，因此可以直接评估；
CNU 标签是从 DDC 映射得到的，因此 CNU 分数同时受到 DDC 预测质量和映射表
合理性的影响。

数据分布也不均衡。有些映射后的 CNU section 有几万条样本，而有些类别非常少
甚至完全不存在。这会影响模型训练，也会影响整体指标的解释。

最后，当前训练数据只覆盖 45 个官方 CNU sections。对于没有出现在弱标签数据中
的 CNU sections，当前监督模型无法直接学习。

## 8. 下一步工作

后续可以继续做的方向包括：

- 进一步检查和优化容易混淆的 DDC-to-CNU 映射。
- 增加 per-class evaluation，尤其关注低频 CNU sections。
- 在较小的人工选择测试集上评估 LLM classifier。
- 在算力允许时继续尝试 transformer-based text representations。
- 在最终报告中加入正确预测和错误预测的具体案例分析。
