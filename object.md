帮我做好以下的框架：

## 目标

使用 **SGD 算法思想** 优化 RLAIF 中的 **JudgePrompt**，使 Judge LLM 对被训练模型 rollout 响应的打分：

* 对齐 **人类打分的绝对值**
* 对齐 **batch 内打分的分布 / 排序结构**

JudgePrompt 被视为一个**可训练但强约束的参数向量**。

---

## 概念映射（修订版）

### 模型权重（Parameters）

* **JudgePrompt（结构化文本）**
* Prompt 被拆分为多个 section（如 Scoring Axis / Anti-Bias / Output Spec）
* 仅允许在指定 section 内做小幅修改，其余视为冻结参数

---

### 前向传播（Forward Pass）

* 使用当前 JudgePrompt 对 rollout response 进行打分
* 单个 response 允许多次打分以估计 **self-consistency variance**

---

### 损失函数（Loss）

Loss 明确拆分为两部分：

1. **绝对值对齐损失**

   * MAE 或 Huber：JudgeScore vs HumanScore

2. **分布 / 排序对齐损失**（修正点）

   * 不使用标准分类交叉熵
   * 使用 **pairwise ranking loss** 或 Kendall / Spearman 的可微 surrogate
   * 仅在 batch 内 top / bottom 分位样本上计算，以降低噪声

总损失：

L = α · MAE + β · RankLoss

---

### Batch Size

* 每次更新从总样本中抽取一个子集
* batch 内需覆盖：

  * 高人类分 / 低人类分
  * 容易混淆的中间区间样本

---

### 梯度

**不直接计算数值梯度**，而是使用强 LLM 构造 **语言空间中的代理梯度**。

#### 梯度样本选择策略（修订）

从当前 batch 中选取以下三类样本，各取 N 个：

1. **打分被显著高估的样本**：

   * JudgeScore ≫ HumanScore
   * 代表 Judge 过度奖励某些特征

2. **打分被显著低估的样本**：

   * JudgeScore ≪ HumanScore
   * 代表 Judge 未识别人类认可的关键信号

3. **打分差距最小的样本**：

   * JudgeScore ≈ HumanScore
   * 用于刻画当前 Prompt 已正确对齐的判据区域

三类样本共同用于构造代理梯度，使梯度同时包含：

* 纠偏信号（高估 / 低估）
* 稳定锚点（已对齐区域）

#### 信息隔离约束（新增，关键）

Gradient Agent **不得接触或推断任何具体被评分样本内容**：

* 不提供原始 response 文本
* 不提供任务类型、领域、上下文
* 不提供单样本级别的具体错误描述

Gradient Agent 仅接收：

* 当前 JudgePrompt
* 聚合后的统计特征（如误差类型计数、方向分布）

该约束用于防止：

* Optimizer 通过梯度侧信道反向拟合训练样本
* JudgePrompt 演化为数据记忆器而非评判器

#### 代理梯度构造（方向化输出，修订）

由于**人类评分仅提供数值而不提供理由**，代理梯度的构造必须避免引入不存在的监督信号。


代理梯度仅由以下三类统计信号构造：

* A 类：**被显著高估的样本**（JudgeScore ≫ HumanScore）

  * 表征 Judge 过度奖励的特征

* B 类：**被显著低估的样本**（JudgeScore ≪ HumanScore）

  * 表征 Judge 未覆盖的人类偏好信号

* C类：**打分差距最小的样本**

  * 即 JudgeScore ≈ HumanScore 的样本
  * 用于分析：当前 Prompt 的“已对齐区域”

Gradient Agent 的输出仅允许是：

* 对上述三类现象的**抽象归因**
* 以及它们在 JudgePrompt 约束层面的含义

不得推断或假设人类评分的具体理由。

代理梯度 = 对 loss 下降方向的符号化指示。

---

### Optimizer（分阶段权限的 Prompt Optimizer Agent）

Optimizer 不直接生成新 JudgePrompt，而是：

* 输入：代理梯度（结构化文本）
* 输出：**提示词修改建议（update direction）**

#### 权限与学习率绑定（修订）

Optimizer 的修改权限与当前学习率（LR）绑定，形成**分阶段搜索空间**：

* ** LR 默认使用余弦退火和warmup调度**

* **当 LR < τ（低学习率阶段）**：

  * 仅允许在既有 editable section 内增减字符
  * 不允许新增或删除 section
  * 主要用于细化 wording、惩罚强度与边界条件

* **当 LR ≥ τ（高学习率阶段）**：

  * 允许增减 editable section
  * 允许对 section 结构做有限重排
  * 主要用于修正早期结构性偏差

除 section 级权限外，仍受字符级 step clipping 约束。

Optimizer 不能：

* 改变总体打分目标
* 引入与既有目标无关的新评判维度

---

### 学习率（Learning Rate，修订版）

学习率通过 **git patch 机制** 实现 step clipping：

* 每次更新：

  * ≤ 指定字符数
  * ≤ 一个 section
* Patch 必须是 unified diff

#### 额外语义约束（新增）

除字符数外，还需限制：

* 情态词等级（can / should / must / always）
* 是否将 soft preference 改为 hard rule

字符长度 + 语义强度 = 实际学习率

---

### 测试集 / 训练集

* 训练集：参与 prompt 优化回路的 response–score 对
* 测试集：完全不参与优化，仅用于监控泛化

人类分数不视为绝对 gold：

* 可选择拟合 mean / rank / annotator 子空间

---

### 早停机制（Early Stopping，修订版）

触发条件：

* Train loss 持续下降
* Val loss 上升或震荡

**附加监控指标**：

* rank correlation 下降
* Judge score entropy 显著降低（collapse）
* self-consistency variance 异常增大

触发后：

* 再执行 n step
* 选取验证集总损失最低的 checkpoint

---

### 防退化机制（新增）

* 显式惩罚 score 分布收缩
* 监控 Judge 对同一输入的打分方差
* 防止 Judge 学会“全部打中间分”

---

### 日志记录与可复现性

* 每一次更新 = 一次 git commit
* commit 中记录：

  * step
  * train / val loss
  * rank corr
  * score entropy
  * 本次修改的代理梯度摘要

JudgePrompt 的演化轨迹即参数训练轨迹。

---

## 总结（不引入新观点）

该设计等价于：

* 在自然语言参数空间中
* 使用强约束、低学习率的近似 SGD
* 优化一个 RLAIF Judge 的代理模型

所有修订仅用于：

* 让梯度有方向
* 让更新可控
* 让退化可观测
