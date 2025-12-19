# SGD Prompt Optimization Framework

基于SGD算法思想的RLAIF Judge Prompt优化框架的Python实现。

## 概述

本框架将JudgePrompt视为可训练参数，使用SGD算法在自然语言参数空间中进行优化，使Judge LLM的打分：
- 对齐人类打分的绝对值
- 对齐batch内打分的分布/排序结构

## 核心特性

### 1. 结构化Prompt管理 (`judge_prompt.py`)
- 将Prompt分为多个section（可编辑/冻结）
- 支持JSON序列化和加载
- 严格的修改权限控制

### 2. 前向传播 (`forward_pass.py`)
- 使用Judge LLM对响应进行打分
- 支持self-consistency variance估计
- 批量打分接口

### 3. 损失函数 (`loss_functions.py`)
实现两类损失：
- **绝对值对齐损失**: MAE或Huber损失
- **排序对齐损失**: Pairwise ranking loss, Kendall τ, Spearman ρ

总损失：`L = α · MAE + β · RankLoss`

### 4. 梯度代理 (`gradient_agent.py`)
使用强LLM构造语言空间中的代理梯度：
- **信息隔离约束**: 不接触具体样本内容
- 仅使用聚合统计信息
- 三类样本选择：高估、低估、已对齐

### 5. 优化器 (`optimizer.py`)
生成Prompt修改建议：
- **学习率绑定权限**:
  - 低LR: 仅修改现有section内容
  - 高LR: 可增删section、重排结构
- 字符级别的step clipping
- 语义约束检查

### 6. 学习率调度 (`lr_scheduler.py`)
- 余弦退火 (Cosine Annealing)
- Warmup机制
- 自适应学习率

### 7. 批量采样 (`batch_sampler.py`)
分层采样策略确保batch覆盖：
- 高人类分样本
- 低人类分样本
- 容易混淆的中间区间样本

### 8. 评估指标 (`metrics.py`)
多维度监控：
- MAE, MSE, RMSE
- Kendall τ, Spearman ρ
- Score entropy (防退化)
- Self-consistency variance

### 9. 早停机制 (`early_stopping.py`)
触发条件：
- 验证集loss上升或震荡
- Rank correlation下降
- Score entropy collapse
- Self-consistency variance异常

### 10. 版本控制 (`version_control.py`)
Git-based prompt evolution tracking：
- 每次更新 = 一次git commit
- 完整的训练轨迹记录
- 支持checkpoint回退

### 11. 训练器 (`trainer.py`)
集成所有组件的完整训练流程：
- 完整的SGD训练循环
- 自动checkpoint管理
- 训练历史记录

## 安装依赖

```bash
pip install numpy scipy
```

## 快速开始

### 1. 创建初始Prompt

```python
from judge_prompt import JudgePrompt

sections = {
    "Scoring Criteria": "Evaluate responses based on quality...",
    "Scale": "Use 1-10 scale...",
    "Output Format": "Output only the numeric score."
}

editable_sections = ["Scoring Criteria"]
prompt = JudgePrompt(sections, editable_sections)
```

### 2. 准备LLM函数

```python
def judge_llm_fn(prompt: str, response: str) -> float:
    # 调用LLM API对response打分
    # 返回数值分数
    pass

def gradient_llm_fn(prompt: str) -> str:
    # 调用强LLM分析统计信息
    # 返回代理梯度文本
    pass

def optimizer_llm_fn(prompt: str) -> str:
    # 调用强LLM生成修改建议
    # 返回修改建议文本
    pass
```

### 3. 准备训练数据

```python
import numpy as np

train_responses = [...]  # 训练集响应列表
train_human_scores = np.array([...])  # 对应的人类打分

val_responses = [...]  # 验证集响应列表
val_human_scores = np.array([...])  # 对应的人类打分
```

### 4. 配置并训练

```python
from trainer import SGDPromptTrainer

config = {
    'max_steps': 100,
    'batch_size': 32,
    'initial_lr': 0.1,
    'min_lr': 0.001,
    'warmup_steps': 10,
    'alpha': 1.0,  # MAE权重
    'beta': 1.0,   # Rank loss权重
    'patience': 5,
}

trainer = SGDPromptTrainer(
    judge_llm_fn=judge_llm_fn,
    gradient_llm_fn=gradient_llm_fn,
    optimizer_llm_fn=optimizer_llm_fn,
    initial_prompt=prompt,
    train_responses=train_responses,
    train_human_scores=train_human_scores,
    val_responses=val_responses,
    val_human_scores=val_human_scores,
    config=config
)

best_prompt = trainer.train()
```

### 5. 保存结果

```python
# 保存最佳prompt
best_prompt.save("best_judge_prompt.json")

# 保存训练历史
trainer.save_history("training_history.json")
```

## 示例

运行完整示例（使用mock LLM函数）：

```bash
python example_usage.py
```

## 架构设计

框架严格遵循object.md中定义的设计原则：

1. **参数**: JudgePrompt（结构化文本）
2. **前向传播**: 使用Prompt对响应打分
3. **损失函数**: 绝对值对齐 + 排序对齐
4. **梯度**: 语言空间代理梯度（信息隔离）
5. **优化器**: 学习率绑定权限的修改生成
6. **学习率**: 余弦退火 + warmup + step clipping
7. **批量采样**: 分层采样确保覆盖
8. **早停**: 多指标监控防退化
9. **日志**: Git-based版本控制

## 文件结构

```
.
├── judge_prompt.py         # Prompt结构表示
├── forward_pass.py         # 前向传播/打分
├── loss_functions.py       # 损失函数实现
├── gradient_agent.py       # 代理梯度构造
├── optimizer.py            # Prompt优化器
├── lr_scheduler.py         # 学习率调度
├── batch_sampler.py        # 批量采样
├── metrics.py              # 评估指标
├── early_stopping.py       # 早停机制
├── version_control.py      # Git版本控制
├── trainer.py              # 主训练器
├── example_usage.py        # 使用示例
└── README.md               # 本文档
```

## 高级用法

### 自定义损失权重

```python
config = {
    'alpha': 2.0,  # 增加MAE权重
    'beta': 0.5,   # 降低Rank loss权重
    'loss_type': 'kendall',  # 使用Kendall τ而非pairwise
}
```

### 调整学习率策略

```python
config = {
    'initial_lr': 0.2,     # 更高的初始学习率
    'min_lr': 0.0001,      # 更低的最小学习率
    'warmup_steps': 20,    # 更长的warmup
}
```

### 启用版本控制

```python
config = {
    'enable_version_control': True,
    'checkpoint_dir': './my_checkpoints',
}
```

## 关键约束

1. **信息隔离**: Gradient Agent不能接触具体样本内容
2. **权限控制**: 修改权限与学习率绑定
3. **Step Clipping**: 字符级别的修改限制
4. **防退化**: 监控score entropy和variance

## 设计理念

本框架将Prompt优化等价为：
> 在自然语言参数空间中，使用强约束、低学习率的近似SGD，优化一个RLAIF Judge的代理模型。

所有设计决策都围绕：
- 让梯度有方向
- 让更新可控
- 让退化可观测

## 许可

MIT License

## 参考

详细设计文档请参阅 `object.md`。
