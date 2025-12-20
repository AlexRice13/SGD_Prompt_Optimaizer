# SGD Prompt Optimization Framework

基于SGD算法思想的RLAIF Judge Prompt优化框架的Python实现。

## 概述

本框架将JudgePrompt视为可训练参数，使用SGD算法在自然语言参数空间中进行优化，使Judge LLM的打分：
- 对齐人类打分的绝对值
- 对齐batch内打分的分布/排序结构

## 核心特性

### 1. 结构化Prompt管理 (`judge_prompt.py`)
- 将Prompt分为多个section（可编辑/冻结）
- 支持JSON序列化和加载（详见 [JudgePrompt格式说明](JUDGE_PROMPT_FORMAT.md)）
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

### 12. 数据集加载器 (`dataset_loader.py`)
从JSONL文件加载训练数据：
- 支持标准JSONL格式：`{"prompt": "...", "response": "...", "score": 8.5}`
- 自动数据集划分（训练/验证）
- 灵活的字段映射配置

### 13. OpenAI集成 (`openai_llm.py`)
OpenAI API集成支持：
- Judge LLM：对响应进行打分
- Gradient Agent：生成代理梯度
- Optimizer：生成修改建议
- 从环境变量读取API密钥和端点

## 安装依赖

```bash
pip install numpy scipy openai
```

## 快速开始

### 1. 设置环境变量

```bash
# 必需：OpenAI API密钥
export OPENAI_API_KEY='your-api-key'

# 可选：自定义API端点
export OPENAI_API_BASE='https://your-custom-endpoint.com/v1'

# 可选：模型选择
export OPENAI_MODEL='gpt-4'

# 可选：初始提示词文件路径
export PROMPT_PATH='/path/to/your/initial_judge_prompt.json'

# 可选：数据集路径
export DATASET_PATH='/path/to/your/dataset.jsonl'
```

### 2. 准备初始JudgePrompt

创建 JSON 格式的 JudgePrompt 文件（详细格式说明见 [JUDGE_PROMPT_FORMAT.md](JUDGE_PROMPT_FORMAT.md)）。

**重要**：sections 数量不限，可以包含任意多个。在优化过程中，当学习率较高时，框架会自动添加或删除 sections。

示例（3个sections）：

```json
{
  "sections": {
    "Scoring Criteria": "根据完整性、清晰度和准确性评估响应。",
    "Scale": "使用 1 到 10 的评分标准，其中 1 表示差，10 表示优秀。",
    "Output Format": "仅输出数字分数，不要输出其他内容。"
  },
  "editable_sections": [
    "Scoring Criteria"
  ]
}
```

你也可以包含更多sections（例如6个或更多）：

```json
{
  "sections": {
    "Scoring Criteria": "...",
    "Anti-Bias": "...",
    "Positive Indicators": "...",
    "Negative Indicators": "...",
    "Scale": "...",
    "Output Format": "..."
  },
  "editable_sections": ["Scoring Criteria", "Anti-Bias", "Positive Indicators", "Negative Indicators"]
}
```

或使用 Python 代码创建：

```python
from judge_prompt import JudgePrompt

prompt = JudgePrompt(
    sections={
        "Scoring Criteria": "你的评分标准...",
        "Scale": "使用 1-10 评分标准",
        "Output Format": "仅输出数字分数",
        # 可以添加任意多个sections
    },
    editable_sections=["Scoring Criteria"]
)
prompt.save("initial_judge_prompt.json")
```

### 3. 准备数据集

创建JSONL格式的数据集文件，每行一个JSON对象：

```jsonl
{"prompt": "Evaluate this response", "response": "This is a high quality answer...", "score": 8.5}
{"prompt": "Evaluate this response", "response": "This is a poor answer", "score": 2.0}
{"prompt": "Evaluate this response", "response": "This is a medium answer...", "score": 5.5}
```

### 4. 使用OpenAI API运行

```python
from judge_prompt import JudgePrompt
from trainer import SGDPromptTrainer
from dataset_loader import DatasetLoader
from openai_llm import create_openai_llm_functions

# 创建初始Prompt
sections = {
    "Scoring Criteria": "Evaluate responses based on quality...",
    "Scale": "Use 1-10 scale...",
    "Output Format": "Output only the numeric score."
}
prompt = JudgePrompt(sections, ["Scoring Criteria"])

# 加载数据集
loader = DatasetLoader()
prompts, responses, scores = loader.load_dataset("dataset.jsonl")
train_resp, train_scores, val_resp, val_scores = loader.split_dataset(
    responses, scores, val_split=0.2
)

# 创建OpenAI LLM函数
judge_fn, gradient_fn, optimizer_fn = create_openai_llm_functions(
    model="gpt-4",
    judge_temperature=0.3,
    gradient_temperature=0.7,
    optimizer_temperature=0.5
)

# 配置并训练
config = {
    'max_steps': 100,
    'batch_size': 32,
    'initial_lr': 0.1,
    'min_lr': 0.001,
    'warmup_steps': 10,
    'alpha': 1.0,
    'beta': 1.0,
    'patience': 5,
}

trainer = SGDPromptTrainer(
    judge_llm_fn=judge_fn,
    gradient_llm_fn=gradient_fn,
    optimizer_llm_fn=optimizer_fn,
    initial_prompt=prompt,
    train_responses=train_resp,
    train_human_scores=train_scores,
    val_responses=val_resp,
    val_human_scores=val_scores,
    config=config
)

best_prompt = trainer.train()
best_prompt.save("best_prompt.json")
```

### 4. 运行示例

使用OpenAI API运行完整示例：

```bash
# 设置API密钥
export OPENAI_API_KEY='your-api-key'

# 运行示例（会自动创建示例数据集）
python example_usage.py
```

如果没有设置API密钥，示例会自动回退到mock函数用于测试。

## 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥（必需） | - |
| `OPENAI_API_BASE` | 自定义API端点（可选） | OpenAI默认端点 |
| `OPENAI_MODEL` | 使用的模型 | `gpt-4` |
| `DATASET_PATH` | 数据集文件路径 | `sample_dataset.jsonl` |
| `MAX_STEPS` | 训练步数 | `10` |
| `BATCH_SIZE` | 批量大小 | `16` |
| `INITIAL_LR` | 初始学习率 | `0.1` |
| `MIN_LR` | 最小学习率 | `0.01` |
| `WARMUP_STEPS` | 预热步数 | `2` |
| `PATIENCE` | 早停耐心值 | `5` |
| `ENABLE_VERSION_CONTROL` | 启用版本控制 | `false` |

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
├── judge_prompt.py            # Prompt结构表示
├── forward_pass.py            # 前向传播/打分
├── loss_functions.py          # 损失函数实现
├── gradient_agent.py          # 代理梯度构造
├── optimizer.py               # Prompt优化器
├── lr_scheduler.py            # 学习率调度
├── batch_sampler.py           # 批量采样
├── metrics.py                 # 评估指标
├── early_stopping.py          # 早停机制
├── version_control.py         # Git版本控制
├── trainer.py                 # 主训练器
├── dataset_loader.py          # 数据集加载器
├── openai_llm.py              # OpenAI API集成
├── example_usage.py           # 使用示例（OpenAI版）
├── example_mock_functions.py  # Mock函数（测试用）
└── README.md                  # 本文档
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
