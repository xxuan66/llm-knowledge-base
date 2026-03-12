# Qwen 系列模型详解

> 通义千问（Qwen）是阿里巴巴通义实验室研发的大语言模型系列，包括 Qwen、Qwen1.5、Qwen2、Qwen2.5、Qwen3 等版本。

## 一、模型家族概览

| 模型 | 发布时间 | 参数量 | 上下文 | 特点 |
|------|---------|--------|--------|------|
| Qwen | 2023.08 | 1.8B/7B/14B/72B | 32K | 初代版本 |
| Qwen1.5 | 2024.02 | 0.5B/1.8B/4B/14B/32B/72B | 32K | 架构升级 |
| Qwen2 | 2024.06 | 0.5B/1.5B/7B/57B/72B | 128K | 长上下文 |
| Qwen2.5 | 2024.09 | 0.5B/3B/7B/14B/32B/72B | 128K | 全面升级 |
| Qwen3 | 2025.Q1 | 待公布 | 256K+ | 多模态 |

## 二、Qwen2.5 架构详解

### 2.1 核心架构参数

**Qwen2.5-7B:**
- 参数量：7B
- 层数：28 层
- 注意力头数：28 Q 头 / 4 KV 头（GQA）
- 隐藏层维度：3584
- 中间层维度（FFN）：18944
- 词表大小：151,936
- 上下文：128K

**Qwen2.5-72B:**
- 参数量：72B
- 层数：80 层
- 注意力头数：64 Q 头 / 8 KV 头（GQA）
- 隐藏层维度：8192
- 中间层维度（FFN）：29568
- 词表大小：151,936
- 上下文：128K

### 2.2 关键技术特点

#### 1. 混合注意力机制

Qwen2.5 采用混合注意力策略：

```python
class HybridAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 前几层使用滑动窗口注意力
        self.sliding_window_layers = config.sliding_window_layers
        # 其余层使用全局注意力
        self.global_layers = config.num_hidden_layers - config.sliding_window_layers
    
    def forward(self, x, layer_idx):
        if layer_idx < self.sliding_window_layers:
            # 滑动窗口：适合局部信息
            return self.sliding_attention(x)
        else:
            # 全局注意力：捕捉长距离依赖
            return self.global_attention(x)
```

**优势：**
- 平衡计算效率和长程建模能力
- 前几层处理局部特征，后几层整合全局信息

#### 2. 长上下文优化

**位置编码：** Qwen2.5 使用改进的 RoPE + YaRN 技术

```python
def yarn_scale_rope(freqs, scale_factor, original_max_pos=4096):
    """YaRN: 动态缩放位置编码"""
    if scale_factor <= 1.0:
        return freqs
    
    # 计算缩放因子
    smooth_factor = 0.1
    scaling_factor = scale_factor
    
    # 应用缩放
    scaled_freqs = freqs / scaling_factor
    
    # 平滑过渡
    mask = torch.arange(len(freqs)) > original_max_pos
    scaled_freqs[mask] *= (1 - smooth_factor) + smooth_factor / scaling_factor
    
    return scaled_freqs
```

**稀疏注意力：** 对于超长序列，使用稀疏注意力减少计算量

```python
def sparse_attention(q, k, v, block_size=256):
    """分块稀疏注意力"""
    bsz, seq_len, dim = q.shape
    
    # 分块
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # 每个 token 只关注相邻块和重要块
    attention_mask = create_sparse_mask(num_blocks, window=3)
    
    # 计算注意力
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores.masked_fill_(attention_mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    
    return torch.matmul(attn, v)
```

#### 3. MoE 架构（Qwen2.5-72B）

Qwen2.5-72B 采用 MoE（Mixture of Experts）架构：

```python
class QwenMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # 64
        self.num_experts_per_token = config.num_experts_per_token  # 4
        
        # 路由器
        self.gate = nn.Linear(config.hidden_size, self.num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            QwenFeedForward(config) for _ in range(self.num_experts)
        ])
    
    def forward(self, x):
        # 计算路由权重
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 选择 top-k 专家
        topk_weights, topk_idx = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1
        )
        
        # 专家计算
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # 找到使用该专家的 token
            token_indices = (topk_idx == expert_idx)
            if token_indices.any():
                expert_output = self.experts[expert_idx](x[token_indices])
                output[token_indices] += (
                    expert_output * topk_weights[token_indices, :].sum(dim=-1, keepdim=True)
                )
        
        return output
```

**MoE 优势：**
- 激活参数仅 7B，但总容量达 72B
- 推理速度接近 7B 模型
- 知识容量大幅提升

#### 4. 多语言支持

Qwen2.5 支持 100+ 语言，词表包含多语言 token：

```python
# 词表分布
# 中文：~30%
# 英文：~50%
# 其他语言：~20%

# 多语言训练数据配比
data_mix = {
    'en': 0.50,
    'zh': 0.30,
    'de': 0.05,
    'fr': 0.03,
    'es': 0.03,
    'ja': 0.03,
    'others': 0.06
}
```

### 2.3 训练数据

**Qwen2.5 训练数据构成：**

| 数据类型 | 占比 | 说明 |
|---------|------|------|
| 网页文本 | 45% | CommonCrawl 等 |
| 书籍 | 15% | 高质量书籍 |
| 代码 | 20% | GitHub、StackOverflow |
| 数学 | 10% | 数学题、证明 |
| 科学论文 | 5% | arXiv 等 |
| 对话数据 | 5% | 多轮对话 |

**训练数据量：** 约 18T tokens

### 2.4 性能基准

**Qwen2.5-72B vs 竞品：**

| 基准 | Qwen2.5-72B | Llama3-70B | Mixtral-8x22B |
|------|------------|-----------|--------------|
| MMLU | 82.3 | 79.5 | 78.2 |
| GSM8K | 89.1 | 84.2 | 81.5 |
| HumanEval | 78.5 | 72.8 | 70.3 |
| MBPP | 82.1 | 78.5 | 76.9 |
| MATH | 56.2 | 48.7 | 45.3 |

### 2.5 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 对话格式
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "解释一下量子力学的基本原理。"}
]

# 应用聊天模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 生成
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2.6 量化部署

```bash
# 使用 Ollama 部署
ollama run qwen2.5:7b

# 使用 vLLM 部署
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 32768

# 使用 llama.cpp（需要转换）
python convert-hf-to-gguf.py Qwen/Qwen2.5-7B-Instruct
./quantize qwen2.5-7b-f16.gguf qwen2.5-7b-q4_k_m.gguf Q4_K_M
```

## 三、Qwen 系列对比

### 3.1 各代改进

| 特性 | Qwen2 | Qwen2.5 | 改进 |
|------|-------|---------|------|
| 上下文 | 128K | 128K | 优化长文本理解 |
| 词表 | 151K | 151K | 优化多语言 |
| 架构 | Dense | MoE(72B) | 容量提升 |
| 代码 | 支持 | 增强 | HumanEval +15% |
| 数学 | 支持 | 增强 | MATH +20% |

### 3.2 推荐场景

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 本地部署 | Qwen2.5-3B | 轻量、速度快 |
| 通用对话 | Qwen2.5-7B | 平衡性能与资源 |
| 专业任务 | Qwen2.5-32B | 更强的推理能力 |
| 企业级 | Qwen2.5-72B | 最佳性能，MoE 高效 |

## 四、相关资源

- **官网：** https://qwenlm.github.io
- **GitHub：** https://github.com/QwenLM/Qwen
- **HuggingFace：** https://huggingface.co/Qwen
- **技术报告：** https://arxiv.org/abs/2407.10671

---

**更新日期：** 2026-03-12
