# JudgePrompt JSON 格式说明

## 概述

JudgePrompt 是框架中用于评分的提示词模板。它可以从 JSON 文件中加载，也可以保存为 JSON 文件。这样可以方便地在不同的优化会话之间重用和共享提示词配置。

**重要**: JudgePrompt 支持**任意数量**的 sections。在优化过程中，当学习率较高时，SGD Agent 可以**动态添加或删除** editable sections。

## JSON 文件格式

JudgePrompt 的 JSON 文件格式如下：

```json
{
  "sections": {
    "Section Name 1": "内容...",
    "Section Name 2": "内容...",
    "Section Name 3": "内容...",
    "...可以有任意多个sections..."
  },
  "meta_sections": [
    "Section Name 2",
    "...列出不可修改的section名称..."
  ]
}
```

### 字段说明

#### 1. `sections` (必需)

一个字典对象，包含提示词的各个部分。每个部分由一个名称（key）和内容（value）组成。

- **类型**: 对象/字典
- **必需**: 是
- **数量限制**: **无限制**，可以包含任意数量的 sections
- **示例**:
  ```json
  "sections": {
    "Scoring Criteria": "根据完整性、清晰度和准确性评估响应。",
    "Anti-Bias": "避免长度偏见。",
    "Scale": "使用 1 到 10 的评分标准，其中 1 表示差，10 表示优秀。",
    "Output Format": "仅输出数字分数，不要输出其他内容。",
    "Examples": "优秀回答示例：...",
    "Special Instructions": "额外的评分指导..."
  }
  ```

**常用的 section 名称**（仅供参考，可以自定义）：
- `Scoring Criteria`: 评分标准
- `Anti-Bias`: 反偏见指导
- `Scale`: 评分量表
- `Output Format`: 输出格式要求
- `Examples`: 示例
- `Special Instructions`: 特殊指导
- 你可以根据需要定义**任何** section 名称和内容

#### 2. `meta_sections` (必需)

一个数组，列出在**任何阶段都不可以修改或删除**的 section 名称。

- **类型**: 数组
- **必需**: 是
- **核心概念**: 
  - `meta_sections` 中的 sections 是**完全冻结**的，永远不能修改或删除
  - **所有不在 `meta_sections` 中的 sections 都自动视为 editable sections**
  - Editable sections 可以根据学习率获得不同的修改权限
- **说明**: 
  - 列出的 section 名称必须存在于 `sections` 中
  - 通常将评分量表 (`Scale`) 和输出格式 (`Output Format`) 设为 meta sections
  - 评分标准等其他 sections 不在 meta_sections 中，因此可编辑
- **示例**:
  ```json
  "meta_sections": [
    "Scale",
    "Output Format"
  ]
  ```

## 动态Section管理

根据 object.md 设计，section 管理规则如下：

### Meta Sections（元 Sections）
- **在任何学习率阶段都不可修改或删除**
- 通常包括：评分量表 (Scale)、输出格式 (Output Format)
- 这些是框架的稳定部分，保持一致性

### Editable Sections（可编辑 Sections）
- **所有不在 `meta_sections` 中的 sections 都是 editable**
- 根据学习率有不同的修改权限：

#### 低学习率阶段（LR < τ）
- **只能**通过 git patch 修改现有 editable sections 的内容
- 修改字数受学习率控制（字数限制 = base_limit × LR）
- **不能**添加新 sections
- **不能**删除 sections

#### 高学习率阶段（LR ≥ τ）
- **可以**通过 git patch 修改现有 editable sections 的内容
- **可以**添加新的 sections（自动成为 editable）
- **可以**删除现有的 editable sections
- **仍然不能**修改或删除 meta_sections

这种设计允许：
- 早期（高LR）：进行结构性探索，添加或删除评分维度
- 后期（低LR）：进行精细调整，优化现有内容的措辞
- 始终保持：核心框架（meta sections）稳定不变

## 完整示例

### 示例 1: 最小配置（2个sections，1个meta）

```json
{
  "sections": {
    "Criteria": "评估质量",
    "Output": "输出分数"
  },
  "meta_sections": ["Output"]
}
```

此配置中：
- `Criteria` 是 editable（可以修改、删除）
- `Output` 是 meta（永远不能修改或删除）

### 示例 2: 标准配置（3个sections，2个meta）

```json
{
  "sections": {
    "Scoring Criteria": "Evaluate responses based on completeness and clarity.",
    "Scale": "Use a scale from 1 to 10, where 1 is poor and 10 is excellent.",
    "Output Format": "Output only the numeric score, nothing else."
  },
  "meta_sections": [
    "Scale",
    "Output Format"
  ]
}
```

此配置中：
- `Scoring Criteria` 是 editable
- `Scale` 和 `Output Format` 是 meta

### 示例 3: 丰富配置（6个sections，2个meta）

```json
{
  "sections": {
    "Scoring Criteria": "根据以下维度评估响应：\n1. 准确性：信息是否正确\n2. 完整性：是否全面回答了问题\n3. 清晰度：表达是否清晰易懂\n4. 相关性：是否紧扣主题",
    "Anti-Bias": "避免基于响应长度进行评分。简洁但准确的回答应该获得高分。",
    "Positive Indicators": "优质回答的特征：\n- 逻辑清晰\n- 证据充分\n- 表达专业",
    "Negative Indicators": "低质回答的特征：\n- 信息错误\n- 逻辑混乱\n- 答非所问",
    "Scale": "使用 1-10 评分标准：\n- 1-3: 差\n- 4-6: 中等\n- 7-8: 良好\n- 9-10: 优秀",
    "Output Format": "仅输出数字分数（1-10 之间的整数或小数），不要输出其他说明。"
  },
  "meta_sections": [
    "Scale",
    "Output Format"
  ]
}
```

此配置中：
- `Scoring Criteria`, `Anti-Bias`, `Positive Indicators`, `Negative Indicators` 都是 editable
- `Scale` 和 `Output Format` 是 meta
- 在优化过程中，可以修改/删除前4个，但不能改后2个

### 示例 4: 无meta配置（所有sections都可编辑）

```json
{
  "sections": {
    "Criteria": "评估标准",
    "Examples": "示例",
    "Notes": "注意事项"
  },
  "meta_sections": []
}
```

此配置中：
- 所有 sections 都是 editable，都可以被修改或删除

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
    meta_sections=["Scale", "Output Format"]  # 这两个不可修改
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
print(f"All sections: {list(prompt.sections.keys())}")
print(f"Meta sections: {prompt.meta_sections}")
print(f"Editable sections: {prompt.get_editable_sections()}")
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
    initial_prompt=initial_prompt,
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

## 向后兼容性

框架支持旧格式的 JSON 文件（使用 `editable_sections` 而非 `meta_sections`）：

```json
{
  "sections": {
    "A": "内容A",
    "B": "内容B",
    "C": "内容C"
  },
  "editable_sections": ["A", "B"]
}
```

加载时会自动转换：
- `editable_sections` = ["A", "B"]
- 自动计算 `meta_sections` = ["C"]（所有不在 editable_sections 中的）

## 设计原则

### 1. Section 划分

将提示词划分为多个 section 的好处：
- **模块化**：每个部分职责明确
- **可控优化**：只优化需要调整的部分（通过 `meta_sections` 控制）
- **结构稳定**：保持固定的框架（meta sections），只优化评分标准

### 2. Meta Sections 与 Editable Sections

**Meta Sections（元 sections）**：
- 完全冻结，任何阶段都不可修改或删除
- 通常是框架性的内容：评分量表、输出格式
- 保证优化过程的稳定性和一致性

**Editable Sections（可编辑 sections）**：
- 所有不在 meta_sections 中的 sections
- 低LR：通过 git patch 修改内容（字数限制）
- 高LR：可以添加新 section、删除现有 section

**动态调整**：在优化过程中，根据学习率：
- **低学习率阶段（LR < τ）**：
  - 只修改 editable sections 的内容
  - 不能添加或删除 sections
  - 保持结构稳定，精细调优
  
- **高学习率阶段（LR ≥ τ）**：
  - 可以修改 editable sections 的内容
  - **可以动态添加新的 sections**
  - **可以删除现有的 editable sections**
  - 可以重排 sections 顺序（可选）
  - 进行结构性探索

**示例**：假设你的初始JSON有3个sections（1个meta，2个editable）：
```json
{
  "sections": {
    "Scoring Criteria": "...",
    "Scale": "...",
    "Output Format": "..."
  },
  "meta_sections": ["Scale", "Output Format"]
}
```

在高LR阶段，优化器可能会：
- 添加 "Anti-Bias" section（成为新的 editable section）
- 添加 "Examples" section
- 删除 "Scoring Criteria"（如果认为不需要）
- 但绝不会删除或修改 "Scale" 和 "Output Format"

结果可能变成4个、5个或更多sections，但 meta sections 始终保持不变。

在低LR阶段，只会精细调整现有的 editable sections 内容。

### 3. 内容编写建议

- **明确具体**：避免模糊的标准
- **可操作性**：标准应该可以被执行
- **避免冲突**：不同 section 之间不应有矛盾
- **语言一致**：使用一致的语言风格
- **Meta vs Editable**：
  - Meta sections: 稳定的框架性内容
  - Editable sections: 需要优化的评分标准

## 常见问题

### Q1: 必须使用英文吗？

不是的。你可以使用任何语言。框架支持 UTF-8 编码，所以中文、日文等都可以正常使用。

### Q2: section 的顺序重要吗？

JSON 对象在 Python 3.7+ 中保持插入顺序。组装完整提示词时，section 会按照字典中的顺序排列。建议按照逻辑顺序排列（如：标准 → 量表 → 格式）。

### Q3: Section数量有限制吗？

**没有限制**！你可以在初始JSON中包含任意数量的sections（从1个到几十个都可以）。在优化过程中，当学习率较高时，优化器还可以动态添加或删除 editable sections。

框架支持的是**动态section管理**，不限制数量。

### Q4: 初始JSON只有3个sections，优化后会不会增加？

**会的**！在高学习率阶段，优化器可以根据需要添加新的 editable sections（例如 Anti-Bias、Examples、Special Instructions等），也可以删除不必要的 editable sections。

最终优化后的prompt可能有5个、10个甚至更多sections，完全取决于优化过程的需要。但 meta sections 始终保持不变。

### Q5: meta_sections 可以为空吗？

可以！如果 `meta_sections` 为空，意味着**所有 sections 都是 editable**，都可以被修改或删除（根据学习率有不同权限）。

这种配置给予优化器最大的自由度。

### Q6: 可以把所有 sections 都设为 meta 吗？

理论上可以，但这样的话：
- 低LR阶段：完全无法修改（因为没有 editable sections）
- 高LR阶段：仍然无法修改 meta sections，但可以添加新的 editable sections

这种配置通常不推荐，因为失去了优化的意义。

### Q7: 旧格式的 JSON 文件还能用吗？

可以！框架完全支持旧格式（使用 `editable_sections`）。加载时会自动转换：
- 读取 `editable_sections`
- 计算 `meta_sections` = 所有 sections - editable_sections
- 创建新格式的 JudgePrompt 对象

保存时会使用新格式（`meta_sections`）。

### Q8: 如何验证 JSON 文件格式正确？

```python
from judge_prompt import JudgePrompt

try:
    prompt = JudgePrompt.load("your_file.json")
    print("✓ JSON 格式正确")
    print(f"  All sections: {list(prompt.sections.keys())}")
    print(f"  Meta sections: {prompt.meta_sections}")
    print(f"  Editable sections: {prompt.get_editable_sections()}")
except Exception as e:
    print(f"✗ JSON 格式错误: {e}")
```

## 参考

- 完整示例请参考：`example_usage.py`
- 使用指南：`USAGE_GUIDE.md`
- 代码文档：`judge_prompt.py`
- 设计文档：`object.md`
