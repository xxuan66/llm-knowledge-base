# 从零搭建 LLM

## 学习目标
学完本节后，你将能够：
- 理解 LLM 的基本组件和实现原理
- 从零开始构建一个简化版的 LLM
- 掌握 PyTorch 实现 Transformer 的方法
- 能够训练和测试自己搭建的模型

## 核心知识点

### 1. LLM 核心组件

#### 1.1 Token Embedding
- **作用**：将 token ID 映射为稠密向量
- **实现**：`nn.Embedding(vocab_size, d_model)`
- **特点**：可学习的参数，捕获语义信息

#### 1.2 Positional Encoding
- **作用**：为模型提供位置信息
- **类型**：可学习的或固定的正弦编码
- **必要性**：自注意力机制本身没有位置信息

#### 1.3 Transformer Layers
- **自注意力层**：捕获序列内部依赖
- **前馈网络**：非线性变换和特征提取
- **残差连接**：缓解梯度消失问题
- **层归一化**：稳定训练过程

#### 1.4 输出层
- **语言模型头**：`nn.Linear(d_model, vocab_size)`
- **输出**：每个位置的词汇表 logits
- **后续**：softmax 得到概率分布

### 2. 模型架构设计

#### 2.1 参数选择
- **d_model**：模型维度（通常 256-1024）
- **n_heads**：注意力头数（通常 4-16）
- **n_layers**：层数（通常 6-12）
- **d_ff**：前馈网络维度（通常 4*d_model）

#### 2.2 超参数影响
- **模型容量**：参数越多，学习能力越强
- **计算成本**：与参数量呈正相关
- **内存需求**：决定可处理的序列长度

### 3. 训练流程

#### 3.1 数据准备
- **分词**：使用 BPE 或 WordPiece
- **格式化**：输入-目标对（如：前文预测下一个词）
- **批处理**：动态批处理，处理变长序列

#### 3.2 训练循环
1. 前向传播：计算模型输出
2. 计算损失：交叉熵损失
3. 反向传播：计算梯度
4. 参数更新：优化器更新
5. 重复直到收敛

#### 3.3 优化技巧
- **学习率调度**：预热+衰减
- **梯度裁剪**：防止梯度爆炸
- **混合精度训练**：加速训练，减少内存

## 代码示例

### 完整 LLM 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换并分割为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 拼接多头并投影
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        return output

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer 编码器块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class MiniLLM(nn.Module):
    """简化版 LLM"""
    
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, 
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, targets=None):
        # 嵌入 + 位置编码
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer 层
        for layer in self.layers:
            x = layer(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=None):
        """生成文本"""
        self.eval()
        generated = input_ids.tolist()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取模型输出
                input_tensor = torch.tensor([generated], device=input_ids.device)
                logits, _ = self.forward(input_tensor)
                
                # 取最后一个位置的 logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k 采样
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个 token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # 停止条件（如遇到结束符）
                if next_token == 0:  # 假设 0 是结束符
                    break
        
        return generated

class SimpleDataset(Dataset):
    """简单文本数据集"""
    
    def __init__(self, text, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # 分词
        tokens = tokenizer.encode(text)
        
        # 创建输入-目标对
        self.data = []
        for i in range(0, len(tokens) - seq_length - 1, seq_length):
            input_ids = tokens[i:i+seq_length]
            targets = tokens[i+1:i+seq_length+1]
            self.data.append((input_ids, targets))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

def train_model(model, dataloader, epochs=10, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            logits, loss = model(input_ids, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

# 使用示例
if __name__ == "__main__":
    # 简化 tokenizer（字符级）
    class CharTokenizer:
        def __init__(self, text):
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            self.vocab_size = len(chars)
        
        def encode(self, text):
            return [self.char_to_idx[ch] for ch in text]
        
        def decode(self, indices):
            return ''.join([self.idx_to_char[idx] for idx in indices])
    
    # 准备数据
    sample_text = """
    机器学习是人工智能的一个分支，它使计算机能够从数据中学习。
    深度学习是机器学习的一个子集，使用神经网络进行学习。
    大语言模型是深度学习的重要应用之一。
    """
    
    tokenizer = CharTokenizer(sample_text)
    dataset = SimpleDataset(sample_text, tokenizer, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建模型
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_len=64
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    trained_model = train_model(model, dataloader, epochs=5, lr=1e-3)
    
    # 测试生成
    test_input = torch.tensor([tokenizer.encode("机器学习")])
    generated = trained_model.generate(test_input, max_length=50, temperature=0.8)
    print(f"生成文本: {tokenizer.decode(generated)}")
```

## 练习题

### 基础题
1. **组件理解**：画出 MiniLLM 的架构图，标注各组件名称和数据流。
2. **参数计算**：计算上述 MiniLLM 模型的参数量（vocab_size=1000, d_model=256, n_heads=4, n_layers=4）。

### 进阶题
3. **代码扩展**：修改代码，添加掩码自注意力机制（用于解码器）。
4. **性能优化**：添加混合精度训练（AMP）支持，比较训练速度差异。

### 思考题
5. **架构改进**：如果要处理更长的序列（如 4096 token），需要对模型做哪些改进？

## 参考资料

### GitHub 热门资源
1. **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)**
   - 相关章节：第 2-4 章《从零构建 LLM》
2. **[Happy-LLM](https://github.com/KMnO4-zx/Happy-LLM)**
   - 相关章节：实践部分《搭建自己的 LLM》
3. **[minGPT](https://github.com/karpathy/minGPT)**
   - 极简 GPT 实现

### 实现参考
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - 最小化 GPT 实现
- [GPT from scratch](https://github.com/karpathy/ng-video-lecture) - Andrej Karpathy 教程

### 工具与库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - 加速训练

### 学习资源
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - GPT-2 图解
- [Building a GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy 视频
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) - Transformer 数学基础