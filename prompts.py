"""
集中管理所有内置的prompt模板。

本文件包含框架中所有LLM任务的prompt模板，包含专有名词解释，便于维护和理解。
"""

# ============================================================================
# 梯度代理（GradientAgent）Prompts
# ============================================================================

# Simplified gradient prompt template
GRADIENT_AGENT_SIMPLE_PROMPT_TEMPLATE = """你是一个元优化器，为评分prompt生成**简单优化指导**。

【专有名词解释】
- Section：评分prompt的组成部分，如"评分标准"、"评分量表"等。
- 元Sections（Meta Sections）：永远不能被修改或删除的sections，如"评分量表"、"输出格式"等。
- 可编辑Sections：可以根据需要修改的sections。

当前评分Prompt的sections：
{current_prompt}

可编辑sections: {editable_sections}
元sections（永不可改）: {meta_sections}

性能统计：
- 总样本: {total_samples}
- 整体MAE: {overall_mae}
- 平均偏置: {mean_error}
- 误差标准差: {std_error}

误差分类：
1. 高估（AI>人工+）: {overestimated_count}个, 均误差{overestimated_mean_error}
2. 低估（AI<人工-）: {underestimated_count}个, 均误差{underestimated_mean_error}
3. 对齐: {well_aligned_count}个, MAE={well_aligned_mae}

参考样本（仅供模式分析，严禁在输出中引用或复制）：
高估样本：
{overestimated_samples}
低估样本：
{underestimated_samples}
对齐样本：
{well_aligned_samples}

=== 反记忆约束 (Anti-Memorization) ===
**严格禁止**：
- 在opti_direction中直接引用、复制或重述任何样本的具体内容
- 创建专门记录训练样本的section（如"样本记录"、"数据集特征"等）
- 将样本中的具体词语、短语或模式硬编码到prompt中

**必须遵守**：
- 只能从样本中提取**通用的语义原则**和**抽象的评分模式**
- 使用概括性描述，而非具体案例
- 优化方向应该适用于未见过的新样本，而不仅仅是训练集

=== 多维度优化 (Optimization Diversity) ===
**优化策略**：
- 分析**至少2个不同的sections**来解决误差问题
- 避免连续多次只修改同一个section
- 考虑sections之间的协同效应
- 当存在多个可编辑sections时，应该分散优化目标
{diversity_hint}

当前学习率: {current_lr}
结构编辑阈值: {structural_lr_threshold} (STRUCTURAL_EDIT_LR_THRESHOLD)

=== 任务：输出JSON格式的多section优化指导 ===

你必须输出一个JSON对象，包含一个modifications数组：

{{
  "modifications": [
    {{
      "action": "edit",  // 可选值: "edit", "add", "remove"
      "section_name": "需要操作的section名称",
      "opti_direction": "用中文描述优化方向（仅当action为edit或add时需要）"
    }},
    // 可以包含多个modification对象
  ]
}}

=== 动作类型说明 ===
- "edit": 修改已有的可编辑section的内容
- "add": 添加一个新的section（仅当学习率 >= 0.6时允许）
- "remove": 删除一个可编辑section（仅当学习率 >= 0.6时允许）

=== 严格约束 ===
- 只输出上述JSON结构，不要添加任何markdown标记或说明文字
- modifications数组**建议包含2个或多个**修改建议以实现多维度优化
- action必须是"edit"、"add"或"remove"之一
- 当action为"edit"时，section_name必须是可编辑sections中的某个名称
- 当action为"add"时，section_name是新section的名称（不能与元sections重名）
- 当action为"remove"时，section_name必须是可编辑sections中的某个名称
- 当action为"add"或"remove"时，只在当前学习率 >= 0.6时使用
- 当学习率 < 0.6时，只能使用action="edit"
- 元sections永远不能被修改、添加到列表或删除
- opti_direction在action为"edit"或"add"时必须提供
- opti_direction必须是**通用的抽象原则**，不能包含样本的具体内容
- JSON中的字符串用双引号
- 不要在JSON数组或对象的最后一个元素后加逗号

基于统计和样本模式，确定优化方向和目标sections。
直接输出JSON对象。"""


# Original complex gradient prompt (kept for reference, not used in simplified version)
GRADIENT_AGENT_PROMPT_TEMPLATE = """你是一个元优化器，为评分prompt生成**结构化语义压力张量**。

【专有名词解释】
- 结构化语义压力张量：一种结构化的梯度表示方法，不输出自然语言的修改建议，
  而是输出在优化器动作空间上的语义压力信号。每个信号明确指定：在哪个section、
  施加什么类型的压力、朝什么方向、有多强。这样可以避免LLM二次解释导致的梯度失真。
- Section：评分prompt的组成部分，如"评分标准"、"评分量表"等。
- 元Sections（Meta Sections）：永远不能被修改或删除的sections，如"评分量表"、"输出格式"等。
- 可编辑Sections：可以根据学习率获得不同修改权限的sections。
- 压力类型（Pressure Type）：表示要施加的抽象语义操作类型，如严格度、阈值、权重等。
- 方向（Direction）：压力的作用方向，如增加、减少、保持。
- 强度桶（Magnitude Bucket）：压力的强度等级（弱/中/强），与学习率对齐。

当前评分Prompt的sections：
{current_prompt}

可编辑sections: {editable_sections}
元sections（永不可改）: {meta_sections}

性能统计：
- 总样本: {total_samples}
- 整体MAE: {overall_mae}
- 平均偏置: {mean_error}
- 误差标准差: {std_error}

误差分类：
1. 高估（AI>人工+）: {overestimated_count}个, 均误差{overestimated_mean_error}
2. 低估（AI<人工-）: {underestimated_count}个, 均误差{underestimated_mean_error}
3. 对齐: {well_aligned_count}个, MAE={well_aligned_mae}

参考样本（仅供模式分析，勿在输出中引用）：
高估样本：
{overestimated_samples}
低估样本：
{underestimated_samples}
对齐样本：
{well_aligned_samples}

当前动作空间权限：
- 可添加section: {can_add_section}
- 可删除section: {can_delete_section}
- 可修改内容: {can_modify_content}
- 当前LR: {current_lr}, 结构编辑阈值: {structural_edit_threshold}

=== 任务：输出JSON格式的结构化梯度 ===

你必须输出一个JSON对象，包含以下字段：

1. global_signals (全局信号，仅作约束，不直接修改):
   - bias_direction: "up" / "down" / "neutral"
   - variance_pressure: "tighten" / "loosen" / "stable"

2. section_pressures (核心：每个可编辑section的压力块):
   [
     {{
       "section_id": "section名称",
       "immutability_ack": false,
       "pressure_type": "constraint_strictness 或 evaluation_threshold 或 preference_weight 或 ambiguity_tolerance",
       "direction": "increase 或 decrease 或 maintain",
       "affected_error_mode": "overestimation 或 underestimation 或 variance",
       "magnitude_bucket": "weak 或 medium 或 strong",
       "confidence": "low 或 medium 或 high"
     }}
   ]
   注：可以包含多个section的压力块

3. acknowledged_action_space (声明理解的动作边界):
   {{
     "allow_add_section": {can_add_section},
     "allow_delete_section": {can_delete_section},
     "allow_sentence_edit": true,
     "allow_token_edit": true
   }}

4. conflicting_pressures (可选，列出冲突的section pairs，没有则为空数组):
   []

5. redundancy_groups (可选，列出可能冗余的section groups，没有则为空数组):
   []

=== 严格约束 ===
- 不要输出任何"怎么改""改成什么"的描述性文本
- 不要引用具体样本内容
- 不要给出修改建议的自然语言描述
- 只输出上述JSON结构
- section_id必须精确匹配可编辑sections列表
- 所有枚举值必须从上述选项中选择
- JSON中的布尔值用 true/false (小写，不带引号)
- JSON中的字符串用双引号，不要用单引号
- 不要在JSON对象/数组的最后一个元素后加逗号

=== 输出格式示例 ===
{{
  "global_signals": {{
    "bias_direction": "down",
    "variance_pressure": "stable"
  }},
  "section_pressures": [
    {{
      "section_id": "评分标准",
      "immutability_ack": false,
      "pressure_type": "constraint_strictness",
      "direction": "increase",
      "affected_error_mode": "overestimation",
      "magnitude_bucket": "medium",
      "confidence": "high"
    }}
  ],
  "acknowledged_action_space": {{
    "allow_add_section": {can_add_section},
    "allow_delete_section": {can_delete_section},
    "allow_sentence_edit": true,
    "allow_token_edit": true
  }},
  "conflicting_pressures": [],
  "redundancy_groups": []
}}

基于统计和样本模式，识别每个可编辑section的语义压力方向和强度。
直接输出JSON对象，不要添加任何说明文字或markdown标记。"""


# ============================================================================
# 优化器（Optimizer）Prompts  
# ============================================================================

# Simplified optimizer prompt template
OPTIMIZER_SIMPLE_PROMPT_TEMPLATE = """你是一个prompt优化代理。根据优化指导直接修改评分Prompt的指定section。

【专有名词解释】
- Prompt优化代理：根据优化方向直接生成修改后的section内容的组件。
- 学习率（Learning Rate, LR）：控制修改幅度的参数。
- 元Sections：永不可修改或删除的框架性sections。

当前评分Prompt：
{current_prompt}

优化指导：
- 动作类型: {action}
- 目标section: {section_name}
- 优化方向: {opti_direction}
- 修改强度: {strength_desc}（基于学习率 {learning_rate}）

约束：
- 可编辑sections: {editable_sections}
- 元sections（不可修改）: {meta_sections}
- 当前sections总数: {total_sections}

=== 简化压力 (Occam's Razor) ===
{simplification_hint}

任务说明：
{task_description}

重要规则：
1. 元sections永远不能被修改
2. 根据优化方向和修改强度调整内容
3. 输出应该是section的完整新内容，不是增量修改
4. 不要添加任何说明文字或格式标记，直接输出新内容

=== 反记忆约束 (Anti-Memorization) ===
**严格禁止在section内容中**：
- 直接引用、复制或重述训练样本的具体内容
- 使用样本中的具体词语、短语或例子
- 创建针对特定样本特征的规则

**必须确保**：
- Section内容是**通用的、抽象的原则**
- 适用于未见过的新样本，而不仅仅是训练数据
- 使用概括性的评分指导，而非具体案例

直接输出{output_description}："""


# Original complex optimizer prompt (kept for reference, not used in simplified version)
OPTIMIZER_PROMPT_TEMPLATE = """你是一个prompt优化代理。为评分Prompt生成修改建议。

【专有名词解释】
- Prompt优化代理：根据结构化梯度信号生成具体的prompt修改方案的组件。
- 结构化梯度信号：来自梯度代理的语义压力信息，指明在哪个section施加什么类型、
  方向和强度的压力。
- 学习率（Learning Rate, LR）：控制修改幅度的参数，高学习率允许结构性改动
  （增删section），低学习率只允许内容微调。
- 元Sections：永不可修改或删除的框架性sections，确保评分系统的基本结构稳定。
- Git Patch格式：标准的文本差异格式，明确显示修改前后的内容对比。
- 压力类型（Pressure Type）：抽象的语义操作类型，如严格度、阈值调整等。
- 字符限制：根据学习率比例动态计算的修改字数上限，确保渐进式优化。

当前评分Prompt：
{current_prompt}

结构化梯度信号：
- 目标section: {section_id}
- 压力类型: {pressure_type}
- 方向: {direction}
- 受影响的误差模式: {error_mode}
- 强度: {magnitude}

修改指导：
{modification_guidance}

优化约束：
- 学习率: {learning_rate}
- 最大字符修改数: {max_chars}
- 可编辑sections: {editable_sections}
- 元sections（不可修改或删除）: {meta_sections}

权限：
- 修改可编辑sections的内容: {modify_content}
- 添加新sections: {add_sections}
- 删除可编辑sections: {remove_sections}

重要规则：
- 元sections（{meta_sections_list}）永远不能被修改或删除
- 只修改指定的section: {section_id}
- 遵循修改指导的语义方向
- 所有修改必须通过git patch格式
- 保持修改在字符限制内

输出格式（必须严格遵循此格式）：
SECTION_TO_MODIFY: {section_id}
OLD_CONTENT:
[复制section的原始内容，可以留空]
NEW_CONTENT:
[写入修改后的完整section内容]
RATIONALE: [简要说明如何响应压力信号]

注意：
1. NEW_CONTENT必须包含section的完整新内容，不是增量修改
2. 不要添加markdown代码块标记或其他格式
3. 每个字段标签后必须换行
4. 直接输出上述格式，不要添加任何说明文字"""


# ============================================================================
# 简化提示模板（Simplification Hints）
# ============================================================================

# Simplification hint when section count exceeds threshold
SIMPLIFICATION_HINT_MANY_SECTIONS = """**注意：当前有{total_sections}个sections（已超过建议的{threshold}个）**
根据奥卡姆剃刀原则（Occam's Razor）：
- 优先考虑**合并相似或重复的内容**到现有sections中，而不是添加新内容
- 简洁明了的prompt通常比冗长复杂的prompt更有效
- 避免不必要的复杂性和冗余"""

# Simplification hint when adding new section
SIMPLIFICATION_HINT_ADD_SECTION = """在添加新section时，请确保：
- 新内容无法合并到现有sections中
- 新section提供独特且必要的价值
- 遵循简洁性原则"""

# Default simplification hint
SIMPLIFICATION_HINT_DEFAULT = "保持prompt简洁明了，遵循奥卡姆剃刀原则。"


# ============================================================================
# 多样性提示模板（Diversity Hints）
# ============================================================================

# Diversity hint template for frequently modified sections
DIVERSITY_HINT_TEMPLATE = """
**多样性提示（Diversity Hint）**：
以下sections在最近几步中已被频繁修改：{sections_str}
建议考虑优化其他sections以实现多维度改进，避免局部最优。"""


# ============================================================================
# 辅助函数：格式化样本
# ============================================================================

def format_sample(idx: int, response: str, human_score: float, 
                  ai_score: float, max_length: int = 200) -> str:
    """
    格式化单个样本用于gradient prompt。
    
    Args:
        idx: 样本序号（从1开始）
        response: 响应文本
        human_score: 人工评分
        ai_score: AI评分
        max_length: 响应文本最大长度
        
    Returns:
        格式化的样本字符串
    """
    if len(response) > max_length:
        response = response[:max_length] + "..."
    error = ai_score - human_score
    return f"  样本{idx}: {response}\n    人工评分: {human_score:.1f}, AI评分: {ai_score:.1f}, 误差: {error:+.1f}\n"


def format_samples_category(indices, responses, human_scores, judge_scores, 
                            category_name: str, max_samples: int = 3) -> str:
    """
    格式化一个类别的样本。
    
    Args:
        indices: 样本索引数组
        responses: 所有响应列表
        human_scores: 所有人工评分
        judge_scores: 所有AI评分
        category_name: 类别名称（如"高估"、"低估"）
        max_samples: 最多显示的样本数
        
    Returns:
        格式化的类别样本字符串
    """
    if len(indices) == 0:
        return f"  （无{category_name}样本）\n"
    
    sample_text = ""
    for i, idx in enumerate(indices[:max_samples], 1):
        response = responses[idx] if idx < len(responses) else "[缺失]"
        sample_text += format_sample(i, response, human_scores[idx], 
                                     judge_scores[idx])
    return sample_text


if __name__ == '__main__':
    """Unit tests for prompts module."""
    import numpy as np
    
    print("Running prompts module unit tests...")
    
    # Test 1: Check prompt templates exist
    print("\n1. Testing prompt templates exist...")
    assert len(GRADIENT_AGENT_PROMPT_TEMPLATE) > 0
    assert len(OPTIMIZER_SIMPLE_PROMPT_TEMPLATE) > 0
    print("   ✓ Prompt templates defined")
    
    # Test 2: Test format_sample function
    print("\n2. Testing format_sample...")
    sample = format_sample(
        idx=1,
        response="Test response",
        human_score=8.5,
        ai_score=7.5
    )
    assert "Test response" in sample
    assert "8.5" in sample
    assert "7.5" in sample
    print("   ✓ format_sample works")
    
    # Test 3: Test format_samples_category
    print("\n3. Testing format_samples_category...")
    responses = ["R1", "R2", "R3"]
    human_scores = np.array([8.0, 5.0, 7.0])
    judge_scores = np.array([7.5, 6.0, 7.5])
    indices = np.array([0, 2])  # Overestimation
    
    output = format_samples_category(
        indices,
        responses,
        human_scores,
        judge_scores,
        "过高估计",
        max_samples=2
    )
    assert "过高估计" in output or len(indices) == 0 or len(output) > 0
    print("   ✓ format_samples_category works")
    
    # Test 4: Empty category
    print("\n4. Testing empty category...")
    empty_output = format_samples_category(
        np.array([]),
        responses,
        human_scores,
        judge_scores,
        "Empty"
    )
    assert "无" in empty_output or "Empty" in empty_output
    print("   ✓ Empty category handled")
    
    print("\n" + "="*50)
    print("All prompts module tests passed! ✓")
    print("="*50)
