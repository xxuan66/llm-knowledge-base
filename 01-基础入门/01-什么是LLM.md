# 什么是大语言模型（LLM）

## 学习目标

学完本节后，你应该能够：

- 用清楚的话解释什么是 LLM
- 说清 LLM 和传统 NLP 模型的差别
- 理解 LLM 为什么既像“模型”，又像“通用能力接口”
- 对后续章节中的 token、上下文窗口、预训练、推理这些概念建立初步认知

## 先给一个直观定义

大语言模型（Large Language Model，LLM）本质上是一类通过海量文本数据训练出来的序列建模系统。它最基础的能力不是“理解世界”，而是**根据已有上下文预测下一个更合理的 token**。但当模型规模、训练数据和训练方法达到一定水平之后，这个简单目标会组合出更复杂的能力，比如问答、总结、改写、代码生成、信息抽取和多轮对话。

所以理解 LLM 时，最重要的不是把它神化成“会思考的机器”，而是先接受一个更稳妥的事实：

> LLM 是一个以语言序列建模为核心、再叠加对齐和工具使用能力的通用系统。

## 为什么它和以前的 NLP 模型不一样

在 LLM 之前，很多 NLP 系统是按任务分别训练的，比如分类模型只做分类、翻译模型只做翻译、命名实体识别模型只做抽取。它们往往性能不错，但迁移性弱、任务边界明确。

LLM 的变化在于，它先做大规模通用预训练，再通过提示词、指令微调、偏好对齐、工具接入等方式适配不同任务。这使得同一个模型可以覆盖大量原本需要单独建模的工作。

| 维度 | 传统 NLP 模型 | LLM |
|------|---------------|-----|
| 训练思路 | 任务定制 | 通用预训练 + 对齐/适配 |
| 数据形态 | 多依赖标注数据 | 大量无标注文本 + 少量高质量指令数据 |
| 使用方式 | 一任务一模型 | 一个模型覆盖多种任务 |
| 交互方式 | 固定输入输出 | 提示词、对话、工具调用 |
| 工程角色 | 模型组件 | 更像能力底座 |

## 一个更实用的理解框架

如果你是从应用角度接触 LLM，可以先把它拆成四层：

1. **分词层**：把文本变成 token
2. **建模层**：用 Transformer 等架构处理序列关系
3. **训练层**：通过预训练、微调、对齐塑造能力边界
4. **使用层**：通过提示词、检索、工具、工作流把能力接入真实任务

后面你会发现，大多数“模型用得好不好”的问题，都能落回这四层中的某一层。

## LLM 的关键特征

### 1. 它是概率系统，不是确定规则系统

LLM 输出的每个 token，本质上都是基于当前上下文下的一次概率选择。即使表现出很强的语言组织能力，它依然可能：

- 说得流畅但不准确
- 结构正确但事实错误
- 逻辑连贯但引用虚构

这就是为什么后续会有评测、对齐、RAG、工具调用这些工程补丁。

### 2. 它的能力很大程度上来自“压缩后的统计经验”

模型并不是把训练语料原文背下来，而是在参数中压缩和抽象了大量语言模式、事实关系、代码结构和任务范式。你可以把它理解为：

> 参数不是知识库本身，而是把大量经验压缩之后得到的一种可调用能力。

### 3. 它越来越像系统的一部分，而不是一个孤立模型

现代 LLM 实际使用时，通常不会单独存在，而是会和下面这些东西结合：

- 检索系统
- 工具调用
- 工作流编排
- 长上下文记忆
- 评测和监控

所以今天学 LLM，不能只盯模型本身，还要理解它在系统里的位置。

## 常见应用场景

LLM 的典型应用已经不只是“聊天”了，常见方向包括：

- 文本生成：写作、润色、改写、摘要
- 知识处理：问答、抽取、分类、标签生成
- 代码开发：补全、解释、重构、调试建议
- 检索增强：企业知识库、文档问答、研究助理
- Agent 场景：调用搜索、数据库、脚本、第三方 API
- 多模态延伸：文本与图像、语音、视频协同理解与生成

## 现在学习 LLM，应该特别注意什么

### 不要只盯模型名字

模型版本会频繁变化，但很多底层认知相对稳定，比如：

- token 是什么
- 上下文窗口为什么重要
- 推理参数如何影响输出
- 为什么需要 RAG
- 为什么需要评测而不是只看 demo

### 不要把“会用”误当成“理解”

只会写提示词，和真正理解 LLM 还差很远。真正的理解至少包括：

- 能判断模型适合什么任务
- 能判断回答为什么不稳定
- 能理解性能、成本、延迟的取舍
- 能知道什么时候该接工具、什么时候该接检索

## 一个简单的判断标准

如果你已经开始能回答下面这些问题，就说明你不只是“会用”，而是真的在理解 LLM：

- 这个问题为什么更适合 RAG，而不是微调？
- 这个回答不稳定，应该先调 prompt、推理参数，还是换模型？
- 这个任务为什么需要工具调用，而不是只靠对话？

## 一个现代化的最小示例

下面这个例子不追求最强模型，而是展示“今天体验开源 LLM”的更现实方式：

```python
from transformers import pipeline

# pip install transformers torch

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "你是一个简洁的 AI 助手。"},
    {"role": "user", "content": "用三句话解释什么是大语言模型。"},
]

output = generator(
    messages,
    max_new_tokens=120,
    temperature=0.7,
)

print(output[0]["generated_text"][-1]["content"])
```

这个例子体现了两个变化：

- 现在很多模型已经默认面向“指令 / 对话”交互
- 学习时应尽量接近真实使用方式，而不是只停留在早期 GPT-2 风格示例

## 代码示例

### 使用 Hugging Face Transformers 快速体验 LLM

```python
from transformers import pipeline

# 创建文本生成管道（需要安装 transformers 库）
# pip install transformers torch

# 使用较小的模型进行演示
generator = pipeline(
    "text-generation",
    model="gpt2",  # 使用开源的 GPT-2 模型
    device=-1  # 使用 CPU，如有 GPU 可改为 0
)

# 生成文本
prompt = "大语言模型是"
output = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

print("输入提示：", prompt)
print("生成结果：", output[0]['generated_text'])
```

### 手动实现简单的文本生成

```python
import torch
import torch.nn.functional as F

class SimpleLLM:
    """简化版的 LLM 概念演示"""
    
    def __init__(self, vocab_size=1000, embed_dim=64):
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size)
    
    def generate(self, input_ids, max_length=20, temperature=1.0):
        """简单的自回归生成"""
        generated = input_ids.tolist()
        
        for _ in range(max_length):
            # 获取最后一个 token 的嵌入
            embed = self.embedding(torch.tensor([generated[-1]]))
            
            # 线性变换得到 logits
            logits = self.linear(embed)
            
            # 应用温度采样
            probs = F.softmax(logits / temperature, dim=-1)
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
        
        return generated

# 使用示例
llm = SimpleLLM()
input_tokens = [1, 2, 3]  # 假设的输入 token 序列
output = llm.generate(input_tokens, max_length=10)
print("生成的 token 序列：", output)
```

## 练习题

### 基础题

1. 用自己的话解释：LLM 为什么不是“普通聊天机器人”的同义词？
2. 传统 NLP 模型和 LLM 的差别，最核心的一点是什么？

### 进阶题

3. 为什么说 LLM 的基础目标只是“预测下一个 token”，但最终却能表现出更复杂的能力？
4. 试着找一个你熟悉的业务场景，判断它更适合“直接调用 LLM”，还是“LLM + 检索 / 工具”。

### 思考题

5. 如果一个模型回答很流畅，但事实经常出错，问题更可能出在模型本身、检索、提示词，还是评测缺失？请给出你的判断。

## 延伸阅读

### GitHub 资源

1. **Happy-LLM**：适合中文入门
2. **LLMs-from-scratch**：适合理解从零实现思路
3. **Awesome-LLM**：适合查资料而不是顺序学习

### 课程与文档

- Stanford CS324
- Hugging Face Learn
- Transformers 官方文档

### 经典论文

- "Attention Is All You Need"
- "Language Models are Few-Shot Learners"

### 阅读建议

- 先用本仓库建立框架，再去读论文
- 先理解问题，再去追具体模型版本
