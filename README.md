# ChemNer
Chinese Chemical Named Entity Recognition

本文针对化学术语构成的领域特征，从语素的角度入手，构建了化学领域语素分类表，并进行了有无语素特征的对比实验。

# CRF-baseline-model

该实验选择当前字的上下文窗口为5，当前输出标签与上一输出标签的关系作为CRF的特征输入，进行模型训练和预测。

基于N-gram特征的CRF识别结果

| 术语 长度 | 术语 总数 | 术语识别数 | 正确识别数 |  正确率 | 召回率 |   F值  |
|:---------:|:---------:|:----------:|:----------:|:-------:|:------:|:------:|
|     1     |     9     |      1     |      1     | 100.00% | 11.11% | 20.00% |
|     2     |     34    |     26     |     23     |  88.46% | 67.65% | 76.67% |
|     3     |    129    |     125    |     116    |  92.80% | 89.92% | 91.34% |
|     4     |    110    |     111    |     103    |  92.79% | 93.64% | 93.21% |
|     5     |     50    |     46     |     44     |  95.65% | 88.00% | 91.67% |
|     6     |     31    |     27     |     23     |  85.19% | 74.19% | 79.31% |
|     7     |     21    |     21     |     20     |  95.24% | 95.24% | 95.24% |
|     8     |     16    |     17     |     15     |  88.24% | 93.75% | 90.91% |
|     9     |     14    |     13     |     12     |  92.31% | 85.71% | 88.89% |
|    >=10   |     34    |     36     |     29     |  80.56% | 85.29% | 82.86% |
|    all    |    448    |     423    |     386    |  91.25% | 86.16% | 88.63% |

> 使用工具[CRF++: Yet Another CRF toolkit](https://github.com/taku910/crfpp)

# BiLSTM-CRF-baseline-model

### 依赖模块

* pytorch=1.13.0
* python3.7+

### 运行方式

1. 运行下列命令，进行模型训练：

   ```python
   python run_lstm_crf.py --do_train
   ```

2. 运行下列命令，进行模型预测

   ```python
   python run_lstm_crf.py --do_predict
   ```

|   Acc  | Recall |   F1   |
|:------:|:------:|:------:|
| 0.9062 | 0.8897 | 0.8979 |

> 修改自[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)

# Hmm-model

把语素和语素类建模为简单稳定的HMM模型，利用改进的前向算法规避术语过长的问题，最终达到了91.58%的较好效果。

# CRF-model

实验选择当前字的上下文窗口为5，当前输出标签与上一输出标签的关系为特征的基础上，加入当前化学语素类的上下文窗口为5作为CRF的特征输入

基于上下文N-gram特征 + 语素类特征的CRF识别结果

| 术语长度 | 术语总数 | 识别出的术语个数 | 正确识别的术语个数 |  正确率 |  召回率 |   F值  |
|:--------:|:--------:|:----------------:|:------------------:|:-------:|:-------:|:------:|
|     1    |     9    |         2        |          1         |  50.00% |  11.11% | 18.18% |
|     2    |    34    |        35        |         31         |  88.57% |  91.18% | 89.86% |
|     3    |    129   |        131       |         123        |  93.89% |  95.35% | 94.62% |
|     4    |    110   |        111       |         108        |  97.30% |  98.18% | 97.74% |
|     5    |    50    |        46        |         44         |  95.65% |  88.00% | 91.67% |
|     6    |    31    |        31        |         28         |  90.32% |  90.32% | 90.32% |
|     7    |    21    |        21        |         20         |  95.24% |  95.24% | 95.24% |
|     8    |    16    |        15        |         15         | 100.00% |  93.75% | 96.77% |
|     9    |    14    |        14        |         13         |  92.86% |  92.86% | 92.86% |
|   >=10   |    34    |        37        |         34         |  91.89% | 100.00% | 95.77% |
|    all   |    448   |        443       |         417        |  94.13% |  93.08% | 93.60% |

> 使用工具[CRF++: Yet Another CRF toolkit](https://github.com/taku910/crfpp)
