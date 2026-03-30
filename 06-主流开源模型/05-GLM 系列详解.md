# GLM 系列详解

> 智谱 AI 研发的 GLM（Generalized Language Model）系列是中国领先的开源大模型，包括 GLM、ChatGLM、GLM-Edge、GLM-130B、GLM-4 等。

GLM 系列最值得关注的地方，在于它长期尝试兼顾中文体验、对话能力、工具调用和本土生态落地。

## 一、模型家族概览

| 模型 | 发布时间 | 参数量 | 上下文 | 特点 |
|------|---------|--------|--------|------|
| GLM-130B | 2022.10 | 130B | 2K | 双语基座 |
| ChatGLM | 2023.03 | 6B | 2K | 轻量对话 |
| ChatGLM2 | 2023.06 | 6B | 32K | 性能提升 |
| ChatGLM3 | 2023.10 | 6B | 128K | 工具调用 |
| GLM-Edge | 2024.01 | 1.8B/9B | 128K | 边缘部署 |
| GLM-4 | 2024.06 |  undisclosed | 128K | 旗舰模型 |

## 先看这个系列适合什么场景

- 中文交互任务
- 面向产品化的对话体验
- 工具调用和应用接入
- 对国内生态支持更敏感的场景

## 二、GLM 架构核心创新

### 2.1 混合架构设计

GLM 采用独特的混合架构，结合 Encoder 和 Decoder 的优势：

```python
class GLMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 双向注意力（Encoder 风格）
        self.bidirectional_attention = SelfAttention(
            config.hidden_size,
            config.num_heads,
            bidirectional=True
        )
        
        # 单向注意力（Decoder 风格）
        self.unidirectional_attention = SelfAttention(
            config.hidden_size,
            config.num_heads,
            bidirectional=False
        )
        
        self.mlp = GLMMLP(config)
    
    def forward(self, x, mask_type="hybrid"):
        # 根据位置决定使用哪种注意力
        if mask_type == "bidirectional":
            # 前缀部分使用双向注意力
            x = x + self.bidirectional_attention(x)
        else:
            # 生成部分使用单向注意力
            x = x + self.unidirectional_attention(x)
        
        x = x + self.mlp(x)
        return x
```

**优势：**
- 理解任务：使用双向注意力，捕捉完整上下文
- 生成任务：使用单向注意力，保证自回归性质
- 灵活切换，适应不同场景

### 2.2 自回归空白填充（AutoRegressive Blank Infilling）

GLM 使用独特的预训练目标：

```python
def glm_pretraining_objective(text):
    """
    GLM 的预训练目标：
    1. 随机 mask 连续 span
    2. 将 mask 部分移到文本末尾
    3. 自回归预测 mask 内容
    """
    # 原始文本
    # "今天 [MASK] 很好"
    
    # 转换为：
    # "今天 很好 [MASK] 天气"
    #              ↑ 需要预测的部分
    
    # 这样既保留了双向上下文信息
    # 又可以使用自回归方式训练
    pass
```

**与 BERT 和 GPT 的对比：**

| 特性 | BERT | GPT | GLM |
|------|------|-----|-----|
| 注意力 | 双向 | 单向 | 混合 |
| 训练目标 | MLM | 自回归 | 空白填充 |
| 擅长任务 | 理解 | 生成 | 两者兼顾 |

### 2.3 ChatGLM 架构参数

**ChatGLM3-6B:**
- 参数量：6.2B
- 层数：28 层
- 注意力头数：32
- 隐藏层维度：4096
- 中间层维度（FFN）：13696
- 词表大小：151,552
- 上下文：128K

## 一个实用判断

如果你做的是中文产品型应用，而不是单纯 benchmark 对比，GLM 系列通常值得作为一条独立候选路线去看。

## 三、ChatGLM 系列详解

### 3.1 ChatGLM3 改进

**相比 ChatGLM2 的改进：**

1. **基座模型增强**
   - 训练数据量增加 3 倍
   - 多语言支持（100+ 语言）
   - 代码能力提升

2. **对齐优化**
   - 更高质量的 SFT 数据
   - 改进的 RLHF 流程
   - 更好的指令遵循

3. **工具调用能力**
   - 原生支持 Function Calling
   - 可调用外部 API
   - 支持多轮工具使用

### 3.2 工具调用示例

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()

# 定义可用工具
tools = [
    {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

# 对话
query = "北京今天天气怎么样？"
response, history = model.chat(tokenizer, query, history=[], tools=tools)

print(response)
# 模型可能输出：
# {"name": "get_weather", "arguments": {"city": "北京"}}
```

### 3.3 多轮对话实现

```python
def multi_turn_chat(model, tokenizer, messages):
    """
    多轮对话
    messages: [{"role": "user", "content": "..."}, 
               {"role": "assistant", "content": "..."},
               ...]
    """
    history = []
    
    for msg in messages:
        if msg["role"] == "user":
            response, history = model.chat(
                tokenizer,
                msg["content"],
                history=history
            )
            print(f"AI: {response}")
        elif msg["role"] == "assistant":
            history.append((msg["content"], None))
    
    return history

# 使用示例
messages = [
    {"role": "user", "content": "你好，介绍一下你自己"},
    {"role": "assistant", "content": "我是 ChatGLM，一个 AI 助手..."},
    {"role": "user", "content": "你能做什么？"}
]

multi_turn_chat(model, tokenizer, messages)
```

## 四、GLM-4 详解

### 4.1 架构升级

GLM-4 在多个方面进行了升级：

| 特性 | ChatGLM3 | GLM-4 | 改进 |
|------|----------|-------|------|
| 上下文 | 128K | 128K | 优化长文本理解 |
| 词表 | 151K | 151K | 优化编码效率 |
| 架构 | Hybrid | Hybrid+ | 混合架构增强 |
| 多模态 | 不支持 | 支持 | 图像理解 |
| 代码 | 支持 | 增强 | 全栈开发能力 |

### 4.2 性能对比

**GLM-4 vs 竞品:**

| 基准 | GLM-4 | GPT-4 | Qwen2-72B | Llama3-70B |
|------|-------|-------|-----------|-----------|
| MMLU | 82.5 | 86.4 | 82.3 | 79.5 |
| GSM8K | 87.2 | 92.0 | 89.1 | 84.2 |
| HumanEval | 76.5 | 87.1 | 78.5 | 72.8 |
| MATH | 54.8 | 62.3 | 56.2 | 48.7 |
| C-Eval | 89.5 | 85.2 | 88.7 | 75.3 |

**注：** GLM-4 在中文任务上表现尤为突出

### 4.3 多模态能力

GLM-4 支持图像理解：

```python
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# 加载多模态模型
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True).cuda()

# 图像问答
image = Image.open("example.jpg").convert("RGB")
query = "这张图片里有什么？"

response, _ = model.chat(tokenizer, query, image=image)
print(response)
```

## 五、GLM-Edge 系列

### 5.1 轻量化设计

GLM-Edge 专为边缘设备设计：

| 模型 | 参数量 | INT4 大小 | 目标设备 |
|------|--------|---------|---------|
| GLM-Edge-1.8B | 1.8B | 1.2 GB | 手机、IoT |
| GLM-Edge-9B | 9B | 6 GB | 笔记本、边缘服务器 |

### 5.2 性能优化

**GLM-Edge 的优化技术：**

1. **知识蒸馏**
   - 从 GLM-4 蒸馏知识
   - 保持 90%+ 的性能
   - 参数量减少 95%

2. **量化感知训练**
   - 训练时模拟 INT4 量化
   - 量化后性能损失 < 2%

3. **算子优化**
   - 针对移动设备优化
   - 支持 GPU、NPU 加速

### 5.3 移动端部署

```python
# 使用 MNN 部署（阿里移动端推理框架）
import MNN.nn as nn

# 加载量化模型
config = nn.Config()
config.backend = 1  # GPU
config.precision = 2  # INT4

model = nn.load_model("glm-edge-1.8b-int4.mnn", config)

# 推理
input_tensor = model.getSessionInput(None)
input_tensor.copyFrom(input_data)
model.runSession()
output = model.getSessionOutput(None).getData()
```

## 六、使用示例

### 6.1 基础对话

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()

# 简单对话
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# 多轮对话
response, history = model.chat(tokenizer, "北京的天气怎么样？", history=history)
print(response)
```

### 6.2 长文档处理

```python
# 128K 上下文处理
long_text = open("long_document.txt").read()

prompt = f"""阅读以下文档并回答问题：

文档内容：
{long_text[:100000]}  # 128K 以内

问题：文档的核心观点是什么？"""

response, _ = model.chat(tokenizer, prompt, history=[])
print(response)
```

### 6.3 代码生成

```python
code_prompt = """用 Python 实现一个快速排序算法，包含详细注释。"""

response, _ = model.chat(tokenizer, code_prompt, history=[])
print(response)
```

## 七、部署建议

### 7.1 硬件需求

| 模型 | FP16 显存 | INT4 显存 | 推荐 GPU |
|------|---------|---------|---------|
| GLM-Edge-1.8B | 3.6 GB | 1.2 GB | 手机/边缘设备 |
| GLM-Edge-9B | 18 GB | 6 GB | RTX 3060 |
| ChatGLM3-6B | 12 GB | 4 GB | RTX 3060 |
| GLM-4 | ~100 GB | ~30 GB | 2×A100 |

### 7.2 部署方式

```bash
# 使用 Ollama
ollama run chatglm3:6b
ollama run glm-4

# 使用 vLLM
python -m vllm.entrypoints.api_server \
    --model THUDM/chatglm3-6b \
    --trust-remote-code \
    --max-model-len 131072

# 使用 llama.cpp（需要转换）
python convert-hf-to-gguf.py THUDM/chatglm3-6b
./quantize chatglm3-6b-f16.gguf chatglm3-6b-q4_k_m.gguf Q4_K_M
```

## 八、选择指南

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 移动端 | GLM-Edge-1.8B | 超轻量，可手机运行 |
| 本地对话 | ChatGLM3-6B | 6B 参数，效果好 |
| 长文档 | ChatGLM3-6B | 128K 上下文 |
| 企业级 | GLM-4 | 最佳性能 |
| 多模态 | GLM-4V | 图像理解 |

## 九、相关资源

- **官网：** https://www.zhipuai.cn
- **GitHub：** https://github.com/THUDM/ChatGLM3
- **HuggingFace：** https://huggingface.co/THUDM
- **技术报告：** https://arxiv.org/abs/2210.02414

---

**更新日期：** 2026-03-12
