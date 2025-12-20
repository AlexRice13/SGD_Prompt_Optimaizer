# JudgePrompt JSON 格式说明

## 概述

JudgePrompt 是框架中用于评分的提示词模板。它可以从 JSON 文件中加载，也可以保存为 JSON 文件。这样可以方便地在不同的优化会话之间重用和共享提示词配置。

## JSON 文件格式

JudgePrompt 的 JSON 文件格式如下：

```json
{
  "sections": {
    "Scoring Criteria": "评分标准的具体内容...",
    "Scale": "评分量表的说明...",
    "Output Format": "输出格式要求..."
  },
  "editable_sections": [
    "Scoring Criteria"
  ]
}
```

### 字段说明

#### 1. `sections` (必需)

一个字典对象，包含提示词的各个部分。每个部分由一个名称（key）和内容（value）组成。

- **类型**: 对象/字典
- **必需**: 是
- **示例**:
  ```json
  "sections": {
    "Scoring Criteria": "根据完整性、清晰度和准确性评估响应。",
    "Scale": "使用 1 到 10 的评分标准，其中 1 表示差，10 表示优秀。",
    "Output Format": "仅输出数字分数，不要输出其他内容。"
  }
  ```

**常用的 section 名称**：
- `Scoring Criteria`: 评分标准
- `Scale`: 评分量表
- `Output Format`: 输出格式要求
- `Anti-Bias`: 反偏见指导
- `Examples`: 示例（可选）

你可以根据需要定义自己的 section 名称和内容。

#### 2. `editable_sections` (必需)

一个数组，列出在优化过程中可以被修改的 section 名称。

- **类型**: 数组
- **必需**: 是
- **说明**: 
  - 列出的 section 名称必须存在于 `sections` 中
  - 未列出的 section 将被视为"冻结"，在优化过程中不会被修改
  - 通常只将核心评分标准设为可编辑，而保持量表和输出格式固定
- **示例**:
  ```json
  "editable_sections": [
    "Scoring Criteria",
    "Anti-Bias"
  ]
  ```

## 完整示例

### 示例 1: 基础配置

```json
{
  "sections": {
    "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
    "Scale": "Use a scale from 1 to 10, where 1 is poor and 10 is excellent.",
    "Output Format": "Output only the numeric score, nothing else."
  },
  "editable_sections": [
    "Scoring Criteria"
  ]
}
```

### 示例 2: 完整配置（包含反偏见）

```json
{
  "sections": {
    "Scoring Criteria": "根据以下维度评估响应：\n1. 准确性：信息是否正确\n2. 完整性：是否全面回答了问题\n3. 清晰度：表达是否清晰易懂",
    "Anti-Bias": "避免基于响应长度进行评分。简洁但准确的回答应该获得高分。",
    "Scale": "使用 1-10 评分标准：\n- 1-3: 差\n- 4-6: 中等\n- 7-8: 良好\n- 9-10: 优秀",
    "Output Format": "仅输出数字分数（1-10 之间的整数或小数），不要输出其他说明。"
  },
  "editable_sections": [
    "Scoring Criteria",
    "Anti-Bias"
  ]
}
```

### 示例 3: 多语言支持

```json
{
  "sections": {
    "Scoring Criteria": "Evaluate based on:\n1. Accuracy of information\n2. Completeness of answer\n3. Clarity of expression\n\n根据以下标准评估：\n1. 信息准确性\n2. 回答完整性\n3. 表达清晰度",
    "Scale": "Use 1-10 scale / 使用 1-10 评分",
    "Output Format": "Output only the numeric score / 仅输出数字分数"
  },
  "editable_sections": [
    "Scoring Criteria"
  ]
}
```

## 使用方法

### 1. 创建 JudgePrompt JSON 文件

你可以手动创建 JSON 文件，或使用 Python 代码生成：

```python
from judge_prompt import JudgePrompt

# 创建 JudgePrompt
prompt = JudgePrompt(
    sections={
        "Scoring Criteria": "你的评分标准...",
        "Scale": "使用 1-10 评分标准",
        "Output Format": "仅输出数字分数"
    },
    editable_sections=["Scoring Criteria"]
)

# 保存为 JSON 文件
prompt.save("my_judge_prompt.json")
```

### 2. 从 JSON 文件加载 JudgePrompt

```python
from judge_prompt import JudgePrompt

# 从文件加载
prompt = JudgePrompt.load("my_judge_prompt.json")

# 查看加载的内容
print(f"Sections: {list(prompt.sections.keys())}")
print(f"Editable: {prompt.editable_sections}")
print(f"Full prompt:\n{prompt.get_full_prompt()}")
```

### 3. 在训练中使用

```python
from trainer import SGDPromptTrainer
from judge_prompt import JudgePrompt

# 加载初始提示词
initial_prompt = JudgePrompt.load("initial_judge_prompt.json")

# 使用加载的提示词进行训练
trainer = SGDPromptTrainer(
    judge_llm_fn=judge_fn,
    gradient_llm_fn=gradient_fn,
    optimizer_llm_fn=optimizer_fn,
    initial_prompt=initial_prompt,  # 使用从文件加载的提示词
    # ... 其他参数
)

best_prompt = trainer.train()

# 保存优化后的提示词
best_prompt.save("optimized_judge_prompt.json")
```

### 4. 使用环境变量

在 `example_usage.py` 中，可以通过环境变量指定 JudgePrompt 文件路径：

```bash
# 设置提示词文件路径
export PROMPT_PATH=/path/to/your/initial_judge_prompt.json

# 运行示例
python example_usage.py
```

如果未设置 `PROMPT_PATH`，程序会在当前目录查找 `initial_judge_prompt.json`，如果不存在则自动创建示例文件。

## 设计原则

### 1. Section 划分

将提示词划分为多个 section 的好处：
- **模块化**：每个部分职责明确
- **可控优化**：只优化需要调整的部分（通过 `editable_sections` 控制）
- **结构稳定**：保持固定的框架（如输出格式），只优化评分标准

### 2. 可编辑性控制

通过 `editable_sections` 控制哪些部分可以被优化：
- **推荐做法**：只将核心评分标准设为可编辑
- **固定部分**：评分量表、输出格式通常应保持不变
- **分阶段优化**：可以在不同阶段调整不同的 section

### 3. 内容编写建议

- **明确具体**：避免模糊的标准
- **可操作性**：标准应该可以被执行
- **避免冲突**：不同 section 之间不应有矛盾
- **语言一致**：使用一致的语言风格

## 常见问题

### Q1: 必须使用英文吗？

不是的。你可以使用任何语言。框架支持 UTF-8 编码，所以中文、日文等都可以正常使用。

### Q2: section 的顺序重要吗？

JSON 对象在 Python 3.7+ 中保持插入顺序。组装完整提示词时，section 会按照字典中的顺序排列。建议按照逻辑顺序排列（如：标准 → 量表 → 格式）。

### Q3: 可以只有一个 section 吗？

理论上可以，但不推荐。建议至少分为"评分标准"和"输出格式"两个部分，这样可以保持输出格式稳定。

### Q4: editable_sections 可以为空吗？

可以，但这样的话提示词在优化过程中不会被修改。这种配置只适合测试或评估现有提示词的场景。

### Q5: 如何验证 JSON 文件格式正确？

```python
from judge_prompt import JudgePrompt

try:
    prompt = JudgePrompt.load("your_file.json")
    print("✓ JSON 格式正确")
    print(f"  Sections: {list(prompt.sections.keys())}")
    print(f"  Editable: {prompt.editable_sections}")
except Exception as e:
    print(f"✗ JSON 格式错误: {e}")
```

## 参考

- 完整示例请参考：`example_usage.py`
- 使用指南：`USAGE_GUIDE.md`
- 代码文档：`judge_prompt.py`
