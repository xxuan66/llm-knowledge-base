# Transformer 架构

## 学习目标

学完本节后，你应该能够：

- 理解 Transformer 为什么会替代 RNN 成为主流架构
- 分清原始 Transformer、Encoder-only、Decoder-only 之间的关系
- 知道多头注意力、前馈网络、位置编码分别解决什么问题
- 用“系统设计”的视角理解它，而不是只背模块名字

## 先回答一个问题：为什么是 Transformer

Transformer 之所以重要，不只是因为它效果好，而是因为它同时解决了大模型扩展最关键的几个问题：

- 序列建模能力强
- 更容易并行训练
- 更适合扩展到大数据和大参数规模
- 更容易成为通用架构底座

从今天往回看，Transformer 真正的历史意义不是“提出了注意力”，而是为大规模语言模型提供了一个可扩展的主干结构。

## 1. 原始 Transformer 在做什么

### 1.1 原始结构
```
输入 → [编码器] → 中间表示 → [解码器] → 输出
```

原始 Transformer 是一个 Encoder-Decoder 结构，最早主要面向机器翻译这类序列到序列任务。

其中：

- Encoder 负责把输入编码成上下文表示
- Decoder 负责在已有输出基础上逐步生成下一个 token

今天的大多数通用 LLM 实际上并不完全沿用这个原始结构，而是更多采用 Decoder-only 变体，但理解原始 Transformer 仍然是必要的。

### 1.2 位置编码

自注意力本身并不知道 token 顺序，因此必须额外引入位置信息。

原始论文使用的是固定的正弦 / 余弦位置编码：

  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```

后续很多模型已经改用 RoPE 等更适合长上下文的方式，但“需要显式或隐式位置表示”这一点没有变。

## 2. 多头注意力到底解决什么问题

### 2.1 计算流程
```
1. 将 Q、K、V 分别线性投影到 h 个头
2. 对每个头独立计算注意力
3. 拼接所有头的输出
4. 线性投影得到最终输出
```

### 2.2 数学表达
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### 2.3 为什么要多头

如果只有一个注意力头，模型只能在一个表示空间里学习一种关注模式。多头机制让模型能从不同子空间同时观察序列关系，比如：

- 有的头更关注局部模式
- 有的头更关注长距离依赖
- 有的头更关注结构对齐

它不只是“并行多做几次计算”，而是显著扩展了模型表达能力。

## 3. 前馈网络不是配角

### 3.1 结构
```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```
它通常是逐位置独立作用的非线性变换模块。虽然注意力最吸引眼球，但 FFN 对模型容量和表达能力同样重要。

在很多现代实现里，FFN 的激活函数和结构会进一步变化，例如 SwiGLU 等，但本质仍然是给注意力输出增加更强的非线性建模能力。

## 4. 残差连接与层归一化为什么关键

### 4.1 残差连接
```
LayerNorm(x + Sublayer(x))
```
它让深层网络更容易训练，也是 Transformer 能不断堆深、堆大的基础条件之一。

### 4.2 层归一化

层归一化帮助稳定训练，避免表示分布在深层网络中持续漂移。

## 5. 从原始 Transformer 到今天的 LLM

学习 Transformer 时，最容易误解的一点是：

> 以为今天的大模型就是把 2017 年原论文原样放大。

实际上并不是。

今天的大模型在很多地方都做了演化，例如：

- 更常见的是 Decoder-only 结构
- 位置编码方式发生变化
- 归一化和残差位置有实现差异
- 注意力优化、KV Cache、GQA / MQA 等推理工程大量发展

所以学习原始 Transformer 的目的，不是记住每个细节，而是理解：

- 为什么自注意力成为主轴
- 为什么这套结构适合大规模扩展
- 为什么后续变体仍然围绕这套思想在演化

## 代码示例

### 完整 Transformer 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, V)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性投影并分割为多头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头并投影
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """Transformer 编码器块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class SimpleTransformer(nn.Module):
    """简化版 Transformer"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 输出层
        output = self.fc_out(x)
        return output

# 使用示例
vocab_size = 10000
model = SimpleTransformer(vocab_size)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试输入
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_ids)
print(f"输出形状: {output.shape}")
```

### 位置编码可视化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_positional_encoding(d_model=128, max_len=100):
    """可视化位置编码"""
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe.T, aspect='auto', cmap='RdBu')
    plt.xlabel('位置')
    plt.ylabel('编码维度')
    plt.title('位置编码可视化')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('positional_encoding.png')
    plt.show()

visualize_positional_encoding()
```

## 练习题

### 基础题
1. **架构理解**：画出 Transformer 的完整架构图，标注各部分名称和数据流向。
2. **参数计算**：计算一个 Transformer 模型（d_model=512, num_heads=8, num_layers=6）的参数量。

### 进阶题
3. **代码实现**：修改上述代码，添加解码器部分，实现完整的 Encoder-Decoder Transformer。
4. **性能分析**：分析为什么 Transformer 比 RNN 更适合并行计算，从计算复杂度角度说明。

### 思考题
5. **架构改进**：如果需要处理超长序列（如 10 万 token），可以对 Transformer 做哪些改进？

### GitHub 热门资源
1. **LLMs-from-scratch**
   - 相关章节：第 4 章《Transformer 架构》
2. **Happy-LLM**
   - 相关章节：第二讲《Transformer 架构详解》
3. **The Annotated Transformer**
   - Transformer 注释版实现

### 经典论文
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer 原始论文
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)

### 可视化资源
- The Illustrated Transformer
- Transformer Visualization
- LLM Visualization

### 实现参考
- Hugging Face Transformers
- PyTorch Transformer Tutorial
- TensorFlow Transformer
