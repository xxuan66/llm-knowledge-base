# Transformer 架构

## 学习目标
学完本节后，你将能够：
- 理解 Transformer 的整体架构设计
- 掌握编码器和解码器的组成结构
- 了解多头注意力机制的实现细节
- 能够实现简化版的 Transformer 模型

## 核心知识点

### 1. 整体架构

#### 1.1 原始 Transformer 结构
```
输入 → [编码器] → 中间表示 → [解码器] → 输出
```

**编码器（Encoder）**：
- N 个相同层堆叠（原论文 N=6）
- 每层包含：多头自注意力 + 前馈神经网络
- 残差连接 + 层归一化

**解码器（Decoder）**：
- N 个相同层堆叠
- 每层包含：掩码多头自注意力 + 编码器-解码器注意力 + 前馈神经网络
- 残差连接 + 层归一化

#### 1.2 位置编码（Positional Encoding）
- **问题**：自注意力没有位置信息
- **解决方案**：添加位置编码向量
- **公式**：
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```

### 2. 多头注意力机制

#### 2.1 计算流程
```
1. 将 Q、K、V 分别线性投影到 h 个头
2. 对每个头独立计算注意力
3. 拼接所有头的输出
4. 线性投影得到最终输出
```

#### 2.2 数学表达
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

#### 2.3 优势
- **并行计算**：多个头可以同时计算
- **多角度关注**：不同头关注不同特征
- **长距离依赖**：直接建模任意位置关系

### 3. 前馈神经网络（FFN）

#### 3.1 结构
```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```
- 两层线性变换
- ReLU 激活函数
- 维度扩展：d_model → 4*d_model → d_model

#### 3.2 作用
- 非线性变换
- 特征提取和转换
- 增加模型容量

### 4. 残差连接与层归一化

#### 4.1 残差连接
```
LayerNorm(x + Sublayer(x))
```
- 解决梯度消失问题
- 允许更深的网络
- 加速训练收敛

#### 4.2 层归一化
- 对每个样本的所有特征进行归一化
- 稳定训练过程
- 允许使用更大的学习率

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

## 参考资料

### GitHub 热门资源
1. **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)**
   - 相关章节：第 4 章《Transformer 架构》
2. **[Happy-LLM](https://github.com/KMnO4-zx/Happy-LLM)**
   - 相关章节：第二讲《Transformer 架构详解》
3. **[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)**
   - Transformer 注释版实现

### 经典论文
- "Attention Is All You Need" (Vaswani et al., 2017) - Transformer 原始论文
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)

### 可视化资源
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Visualization](https://github.com/jessevig/bertviz)
- [LLM Visualization](https://bbycroft.net/llm)

### 实现参考
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [TensorFlow Transformer](https://www.tensorflow.org/text/tutorials/transformer)