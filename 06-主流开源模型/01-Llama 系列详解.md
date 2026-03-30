# Llama 系列模型详解

> Meta 开源的 Llama 系列是当前最主流的开源大语言模型，包括 Llama、Llama2、Llama3、Llama3.1、Llama3.2 等版本。

如果你只想先抓住一句话：Llama 系列最大的价值，不只是模型本身，而是它长期扮演了开源生态基线模型的角色。

## 一、模型家族概览

| 模型 | 发布时间 | 参数量 | 上下文 | 开源协议 |
|------|---------|--------|--------|---------|
| Llama | 2023.02 | 7B/13B/33B/65B | 2K | 研究许可 |
| Llama2 | 2023.07 | 7B/13B/70B | 4K | 可商用 |
| Llama3 | 2024.04 | 8B/70B | 8K | 可商用 |
| Llama3.1 | 2024.07 | 8B/70B/405B | 128K | 可商用 |
| Llama3.2 | 2024.09 | 1B/3B/11B/90B | 128K | 可商用 |

## 先看这个系列适合什么人

- 想要生态成熟、资料丰富、部署经验多的模型使用者
- 想做本地部署和工程实验的开发者
- 想把模型当作基线进行横向对比的人

## 二、Llama3 架构详解

### 2.1 核心架构参数

**Llama3-8B:**
- 参数量：8B
- 层数：32 层
- 注意力头数：32 Q 头 / 8 KV 头（GQA）
- 隐藏层维度：4096
- 中间层维度（FFN）：14336
- 词表大小：128,256

**Llama3-70B:**
- 参数量：70B
- 层数：80 层
- 注意力头数：64 Q 头 / 8 KV 头（GQA）
- 隐藏层维度：8192
- 中间层维度（FFN）：28672
- 词表大小：128,256

### 2.2 关键技术改进

#### 1. Grouped Query Attention (GQA)

**原理：** 多个查询头共享一组 KV 头，减少 KV Cache 大小。

```
传统 MHA:  Q[32] → K[32], V[32]  (32 组 KV)
GQA:      Q[32] → K[8],  V[8]   (8 组 KV, 每组服务 4 个 Q 头)
```

**优势：**
- KV Cache 减少 75%（8B 模型）
- 推理速度提升 2-3 倍
- 几乎不影响模型质量

**代码实现：**
```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        
        # 重复 KV 以匹配 Q 头数
        k = k.repeat_interleave(self.num_kv_groups, dim=2)
        v = v.repeat_interleave(self.num_kv_groups, dim=2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return self.o_proj(output.flatten(2))
```

#### 2. RoPE 位置编码改进

Llama3 扩展了 RoPE 的频率范围，支持更长上下文：

```python
def compute_rope_freqs(max_seq_len, dim, theta=500000.0):
    """Llama3 使用更大的 theta 值"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# Llama3 使用 theta=500000（Llama2 是 10000）
# 这有助于更好地处理长序列
```

#### 3. SwiGLU 激活函数

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        # 将输入分成两部分
        x1, x2 = x.chunk(2, dim=-1)
        # Swish(x1) * x2
        return F.silu(x1) * x2

# 在 FFN 中的应用
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

#### 4. 词表扩展

Llama3 将词表从 Llama2 的 32K 扩展到 128K：

**优势：**
- 更高的压缩率（更少的 token 表示相同内容）
- 更好的多语言支持
- 减少 OOV（未登录词）

**Tokenizer 配置：**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print(f"词表大小：{len(tokenizer)}")  # 128256

# 特殊 token
print(tokenizer.all_special_tokens)
# ['
