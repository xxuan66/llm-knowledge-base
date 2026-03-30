# DeepSeek 系列模型详解

> 深度求索（DeepSeek）是专注于 AGI 研究的中国公司，其开源模型以高性价比著称，包括 DeepSeek-LLM、DeepSeek-Coder、DeepSeek-V2、DeepSeek-V3 等。

DeepSeek 系列最值得关注的地方，不只是模型能力本身，而是它在架构和推理效率上经常带来很有辨识度的工程创新。

## 一、模型家族概览

| 模型 | 发布时间 | 参数量 | 激活参数 | 上下文 | 特点 |
|------|---------|--------|---------|--------|------|
| DeepSeek-LLM | 2023.11 | 7B/67B | - | 4K | 初代版本 |
| DeepSeek-Coder | 2024.01 | 1B/7B/33B | - | 16K | 代码专用 |
| DeepSeek-V2 | 2024.05 | 236B | 21B | 128K | MoE 架构 |
| DeepSeek-V2.5 | 2024.09 | 236B | 21B | 128K | 全面升级 |
| DeepSeek-V3 | 2024.12 | 671B | 37B | 256K | 大模型版本 |

## 先看这个系列为什么重要

这个系列值得重点关注的原因包括：

- 对代码和推理任务有持续投入
- 在 MoE 和推理优化方向上有明显辨识度
- 经常能把“高能力”和“高性价比”一起往前推

## 二、DeepSeek-V2 架构详解

### 2.1 核心创新：MLA（Multi-Head Latent Attention）

DeepSeek-V2 引入了 MLA 注意力机制，大幅降低 KV Cache 大小：

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # 压缩 KV
        self.kv_compression = nn.Linear(
            self.num_heads * self.head_dim,
            config.kv_compression_dim  # 512
        )
        
        # 解压 KV
        self.kv_decompression = nn.Linear(
            config.kv_compression_dim,
            self.num_heads * self.head_dim
        )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
    
    def forward(self, x, past_kv=None):
        bsz, seq_len, _ = x.shape
        
        # Q 投影
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        # KV 压缩
        kv_input = x  # 或者使用 RoPE 后的 Q
        compressed_kv = self.kv_compression(kv_input)
        
        # 存储压缩的 KV（节省显存）
        if past_kv is not None:
            compressed_kv = torch.cat([past_kv, compressed_kv], dim=1)
        
        # KV 解压
        kv = self.kv_decompression(compressed_kv)
        k = kv.view(bsz, -1, self.num_heads, self.head_dim)
        v = kv.view(bsz, -1, self.num_heads, self.head_dim)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).flatten(2)
        
        return self.o_proj(output), compressed_kv
```

**MLA 优势：**
- KV Cache 减少 93%（从 27KB/token 降到 1.7KB/token）
- 支持更长上下文（128K）
- 推理速度提升 3 倍

### 2.2 架构参数

**DeepSeek-V2:**
- 总参数：236B
- 激活参数：21B
- 层数：60 层
- 注意力头数：128 Q 头 / 128 KV 头（MLA 压缩后等效）
- 隐藏层维度：5120
- 中间层维度（FFN）：15360（MoE：16 专家，top-2）
- 词表大小：102,400
- 上下文：128K

**DeepSeek-V3:**
- 总参数：671B
- 激活参数：37B
- 层数：61 层
- 注意力头数：128 Q 头 / 128 KV 头（MLA）
- 隐藏层维度：7168
- 中间层维度（FFN）：20480（MoE：256 专家，top-8）
- 词表大小：129,216
- 上下文：256K

### 2.3 MoE 架构升级

DeepSeek-V2/V3 采用改进的 MoE 设计：

```python
class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # V2: 16, V3: 256
        self.top_k = config.top_k  # V2: 2, V3: 8
        self.num_shared_experts = 2  # 共享专家
        
        # 路由器
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # 专家列表
        self.experts = nn.ModuleList([
            DeepSeekFeedForward(config)
            for _ in range(self.num_experts)
        ])
        
        # 共享专家（所有 token 都使用）
        self.shared_experts = nn.ModuleList([
            DeepSeekFeedForward(config)
            for _ in range(self.num_shared_experts)
        ])
    
    def forward(self, hidden_states):
        # 路由决策
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 选择 top-k 专家
        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 专家计算
        output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            token_indices = (topk_idx == expert_idx)
            if token_indices.any():
                expert_output = self.experts[expert_idx](hidden_states[token_indices])
                output[token_indices] += (
                    expert_output * topk_weights[token_indices, :].sum(dim=-1, keepdim=True)
                )
        
        # 共享专家（始终激活）
        for shared_expert in self.shared_experts:
            output = output + shared_expert(hidden_states)
        
        return output
```

**共享专家的作用：**
- 保证基础能力（所有 token 共享）
- 路由专家处理特定领域知识
- 提升模型稳定性和泛化能力

### 2.4 多 Token 预测技术

DeepSeek-V3 引入了多 token 预测（Multi-Token Prediction）：

```python
class MultiTokenPrediction(nn.Module):
    def __init__(self, config, num_predict_tokens=3):
        super().__init__()
        self.num_predict_tokens = num_predict_tokens
        
        # 多个预测头
        self.prediction_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size)
            for _ in range(num_predict_tokens)
        ])
    
    def forward(self, hidden_states):
        # 并行预测多个 token
        predictions = []
        for i, head in enumerate(self.prediction_heads):
            # 第 i 个头预测第 t+i 个 token
            pred = head(hidden_states)
            predictions.append(pred)
        
        return predictions

# 训练时使用
def multi_token_loss(predictions, targets):
    """多 token 联合损失"""
    total_loss = 0
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # 第 i 个头的目标是第 t+i 个 token
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
        # 越远的 token 权重越低
        weight = 1.0 / (i + 1)
        total_loss += weight * loss
    
    return total_loss / len(predictions)
```

**优势：**
- 推理速度提升 2-3 倍
- 一次前向传播预测多个 token
- 适合低延迟场景

## 一个实用判断

如果你的任务很看重代码、推理效率或大容量模型在成本上的可行性，DeepSeek 系列很值得重点比较。

## 三、DeepSeek-Coder 详解

### 3.1 代码能力强化

DeepSeek-Coder 在代码任务上表现优异：

| 基准 | DeepSeek-Coder-33B | StarCoder-15B | CodeLlama-34B |
|------|-------------------|--------------|--------------|
| HumanEval | 78.0 | 60.4 | 72.6 |
| MBPP | 80.5 | 69.5 | 76.2 |
| MultiPL-E | 65.3 | 51.2 | 60.8 |

### 3.2 训练数据

**DeepSeek-Coder 训练数据构成：**

| 数据类型 | 占比 | 说明 |
|---------|------|------|
| GitHub 代码 | 70% | 多编程语言 |
| 技术文档 | 15% | StackOverflow、官方文档 |
| 代码注释 | 10% | 高质量注释 |
| 编程书籍 | 5% | 编程教材 |

**支持语言：** Python、Java、C++、JavaScript、Go、Rust、等 80+ 种

### 3.3 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载代码模型
model_name = "deepseek-ai/deepseek-coder-33b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 代码补全
prompt = """# Python 快速排序
def quicksort(arr):
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False  # 代码生成通常用 greedy decoding
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 四、性能对比

### 4.1 DeepSeek-V2.5 vs 竞品

| 基准 | DeepSeek-V2.5 | Mixtral 8x22B | Llama3-70B | Qwen2-72B |
|------|--------------|--------------|-----------|----------|
| MMLU | 80.5 | 78.2 | 79.5 | 82.3 |
| GSM8K | 85.3 | 81.5 | 84.2 | 89.1 |
| HumanEval | 75.8 | 70.3 | 72.8 | 78.5 |
| MATH | 52.1 | 45.3 | 48.7 | 56.2 |
| MBPP | 79.2 | 76.9 | 78.5 | 82.1 |

### 4.2 推理性能

**DeepSeek-V2 vs Llama3-70B（A100 80G）:**

| 指标 | DeepSeek-V2 | Llama3-70B | 优势 |
|------|-------------|-----------|------|
| 显存占用 | 48 GB | 140 GB | 66% 减少 |
| 推理速度 | 85 tok/s | 25 tok/s | 3.4 倍快 |
| 首 token 延迟 | 50 ms | 120 ms | 58% 减少 |

## 五、部署建议

### 5.1 硬件需求

| 模型 | FP16 显存 | INT4 显存 | 推荐 GPU |
|------|---------|---------|---------|
| DeepSeek-Coder-7B | 14 GB | 5 GB | RTX 3060 12G |
| DeepSeek-V2 | 468 GB | 120 GB | 8×A100 |
| DeepSeek-V2 (INT4) | 120 GB | - | 2×A100 80G |

### 5.2 部署方式

```bash
# 使用 Ollama（推荐）
ollama run deepseek-coder:6.7b
ollama run deepseek-v2

# 使用 vLLM
python -m vllm.entrypoints.api_server \
    --model deepseek-ai/deepseek-coder-33b-instruct \
    --tensor-parallel-size 2 \
    --max-model-len 16384

# 使用 llama.cpp
python convert-hf-to-gguf.py deepseek-ai/deepseek-coder-7b-base
./quantize deepseek-coder-7b-f16.gguf deepseek-coder-7b-q4_k_m.gguf Q4_K_M
```

## 六、选择指南

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 代码补全 | DeepSeek-Coder-7B | 轻量、代码能力强 |
| 代码生成 | DeepSeek-Coder-33B | 最佳代码能力 |
| 通用对话 | DeepSeek-V2.5 | 综合能力强 |
| 长文档处理 | DeepSeek-V2.5 | 128K 上下文 |
| 企业级 | DeepSeek-V3 | 旗舰性能 |

## 七、相关资源

- **官网：** https://www.deepseek.com
- **GitHub：** https://github.com/deepseek-ai
- **HuggingFace：** https://huggingface.co/deepseek-ai
- **技术报告：** https://arxiv.org/abs/2405.04434

---

**更新日期：** 2026-03-30
