# Tokenizer 实现

## 学习目标

学完本节后，你应该能够：

- 理解 tokenizer 为什么是 LLM 系统的基础环节
- 知道 BPE / WordPiece / SentencePiece 的差异
- 理解 tokenizer 不只是“分词”，还会影响成本、上下文和模型能力边界
- 从教学角度实现一个简化 tokenizer

## 先说一个常被低估的事实

很多人学 LLM 时会把 tokenizer 当成“前处理细节”，但实际上它会直接影响：

- token 数量
- 上下文利用率
- 多语言表现
- 代码和特殊符号处理效果
- 训练和推理成本

所以 tokenizer 不是模型外的附属组件，而是模型能力的一部分。

## 核心知识点

### 1. Tokenizer 基础

#### 1.1 什么是 Tokenizer
Tokenizer 是将原始文本转换为模型可以处理的数字序列的工具，是 LLM 的第一个处理步骤。

更准确一点说，它负责把文本映射为一套稳定的离散符号表示，并保证训练和推理阶段使用同一套规则。

**核心功能**：
- **编码（Encode）**：文本 → token ID 序列
- **解码（Decode）**：token ID 序列 → 文本
- **词汇表管理**：维护 token 到 ID 的映射

#### 1.2 Tokenization 方法

**字符级（Character-level）**：
- **优点**：词汇表小（几十到几百），无 OOV 问题
- **缺点**：序列长，语义信息少
- **适用**：简单任务，小模型

**词级（Word-level）**：
- **优点**：保留完整词义
- **缺点**：词汇表巨大，OOV 问题严重
- **适用**：传统 NLP 任务

**子词级（Subword-level）**：
- **优点**：平衡词汇表大小和语义表达
- **代表**：BPE、WordPiece、SentencePiece
- **适用**：现代 LLM，解决 OOV 问题

### 2. 为什么现代模型普遍使用子词级 tokenizer

字符级太细，序列会变长；词级太粗，OOV 问题严重。子词级 tokenizer 正好提供了更现实的折中。

### 3. BPE（字节对编码）算法

#### 2.1 算法原理
BPE 是一种数据压缩算法，通过迭代合并最频繁的字符对来构建子词词汇表。

**步骤**：
1. 初始化词汇表：所有单个字符
2. 统计所有相邻 token 对的频率
3. 合并频率最高的 token 对
4. 重复步骤 2-3，直到达到目标词汇表大小

#### 2.2 算法伪代码
```
初始化词汇表 V = {所有字符}
for i in range(目标词汇表大小 - 初始词汇表大小):
    统计所有相邻 token 对的频率
    找到频率最高的 token 对 (A, B)
    将新 token AB 添加到 V
    将所有文本中的 A, B 替换为 AB
```

#### 2.3 编码和解码
**编码**：
1. 将文本分割为字符
2. 贪心地合并最长匹配的子词

**解码**：
1. 将 token ID 转换为子词
2. 拼接子词得到完整文本

### 4. Tokenizer 类型对比

| 类型 | 代表 | 词汇表大小 | 优点 | 缺点 |
|------|------|-----------|------|------|
| **BPE** | GPT-2, GPT-3 | 50K-100K | 简单高效 | 依赖合并顺序 |
| **WordPiece** | BERT | 30K-50K | 基于似然 | 需要概率计算 |
| **SentencePiece** | T5, LLaMA | 32K-64K | 语言无关 | 需要训练 |
| **Byte-level BPE** | GPT-2 | 50K | 处理任意文本 | 序列较长 |

### 5. 现代 Tokenizer 特点

#### 4.1 特殊 Token
- **[PAD]**：填充 token，用于批处理
- **[UNK]**：未知 token，处理 OOV
- **[CLS]**：分类 token，用于 BERT
- **[SEP]**：分隔 token，分隔句子
- **[MASK]**：掩码 token，用于 MLM

#### 4.2 多语言支持
- **统一词汇表**：包含多种语言字符
- **语言特定 token**：常见词汇的完整词
- **字符覆盖**：确保所有语言字符被包含

## 代码实现之外，还该关注什么

如果你在真实项目里选 tokenizer，建议重点关注：

1. 同样文本会切成多少 token
2. 中文、英文、代码混合时表现如何
3. 特殊符号和格式化文本是否稳定
4. 是否容易和目标模型生态兼容

## 代码示例

### BPE Tokenizer 完整实现

```python
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class BPETokenizer:
    """BPE Tokenizer 实现"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # 特殊 token
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
    
    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """统计相邻 token 对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """合并词汇表中的 token 对"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            w_out = pattern.sub(''.join(pair), word)
            new_vocab[w_out] = vocab[word]
        
        return new_vocab
    
    def train(self, texts: List[str]):
        """训练 BPE Tokenizer"""
        # 初始化词汇表（字符级）
        vocab = defaultdict(int)
        for text in texts:
            # 分词为字符，添加空格标记
            words = text.strip().split()
            for word in words:
                # 每个字符用空格分隔
                vocab[' '.join(list(word)) + ' </w>'] += 1
        
        # 初始化特殊 token
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.id_to_token[idx] = token
        
        # 迭代合并
        num_merges = self.vocab_size - len(self.special_tokens)
        
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # 找到最高频的 token 对
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            
            # 记录合并规则
            self.merges.append(best)
            
            # 添加新 token 到词汇表
            new_token = ''.join(best)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.id_to_token[len(self.vocab) - 1] = new_token
            
            if (i + 1) % 100 == 0:
                print(f"Merged {i+1}/{num_merges} pairs")
        
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """将单词分割为子词"""
        if word in self.vocab:
            return [word]
        
        # 分割为字符
        word = ' '.join(list(word)) + ' </w>'
        symbols = word.split()
        
        # 应用合并规则
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            
            # 找到可以合并的最高优先级对
            bigram = None
            for pair in pairs:
                if pair in self.merges:
                    bigram = pair
                    break
            
            if bigram is None:
                break
            
            # 合并
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            
            symbols = new_symbols
        
        # 移除 </w> 标记
        if symbols and symbols[-1].endswith('</w>'):
            symbols[-1] = symbols[-1].replace('</w>', '')
        
        return symbols
    
    def encode(self, text: str) -> List[int]:
        """编码文本为 token ID 序列"""
        tokens = []
        words = text.strip().split()
        
        for word in words:
            subwords = self._tokenize_word(word)
            for subword in subwords:
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                else:
                    tokens.append(self.special_tokens["[UNK]"])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """解码 token ID 序列为文本"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append("[UNK]")
        
        # 合并子词，处理空格
        text = ''
        for token in tokens:
            if token.endswith('</w>'):
                text += token[:-4] + ' '
            else:
                text += token
        
        return text.strip()

# 测试 BPE Tokenizer
if __name__ == "__main__":
    # 训练数据
    training_texts = [
        "机器学习是人工智能的一个重要分支",
        "深度学习是机器学习的一个子集",
        "大语言模型是深度学习的重要应用",
        "自然语言处理是人工智能的核心领域",
        "Transformer 架构改变了自然语言处理的发展方向"
    ]
    
    # 创建并训练 tokenizer
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.train(training_texts)
    
    # 测试编码
    test_text = "机器学习改变了人工智能"
    encoded = tokenizer.encode(test_text)
    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    
    # 测试解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")
    
    # 显示部分词汇表
    print(f"\n词汇表大小: {len(tokenizer.vocab)}")
    print("部分词汇表:")
    for i, (token, idx) in enumerate(list(tokenizer.vocab.items())[:20]):
        print(f"  {idx}: '{token}'")
```

### SentencePiece 简化实现

```python
import sentencepiece as spm
import tempfile
import os

def train_sentencepiece_tokenizer(texts, vocab_size=1000, model_prefix='spm'):
    """训练 SentencePiece Tokenizer"""
    
    # 将文本写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for text in texts:
            f.write(text + '\n')
        temp_file = f.name
    
    try:
        # 训练 SentencePiece 模型
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            user_defined_symbols=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        )
        
        # 加载模型
        sp = spm.SentencePieceProcessor()
        sp.load(f'{model_prefix}.model')
        
        return sp
    
    finally:
        # 清理临时文件
        os.unlink(temp_file)

# 测试 SentencePiece
if __name__ == "__main__":
    training_texts = [
        "机器学习是人工智能的一个重要分支",
        "深度学习是机器学习的一个子集",
        "大语言模型是深度学习的重要应用",
        "自然语言处理是人工智能的核心领域",
        "Transformer 架构改变了自然语言处理的发展方向"
    ]
    
    sp = train_sentencepiece_tokenizer(training_texts, vocab_size=200)
    
    # 测试编码解码
    test_text = "机器学习改变了人工智能"
    
    # 编码
    encoded = sp.encode(test_text, out_type=int)
    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    
    # 解码
    decoded = sp.decode(encoded)
    print(f"解码结果: {decoded}")
    
    # 显示词汇表
    print(f"\n词汇表大小: {sp.get_piece_size()}")
    print("部分词汇表:")
    for i in range(min(20, sp.get_piece_size())):
        print(f"  {i}: '{sp.id_to_piece(i)}'")
```

### Tokenizer 比较工具

```python
from transformers import AutoTokenizer

class TokenizerComparator:
    """比较不同 Tokenizer 的性能"""
    
    def __init__(self):
        self.tokenizers = {}
    
    def load_tokenizers(self):
        """加载不同类型的 tokenizer"""
        self.tokenizers = {
            'BPE': AutoTokenizer.from_pretrained('gpt2'),
            'WordPiece': AutoTokenizer.from_pretrained('bert-base-uncased'),
            'SentencePiece': AutoTokenizer.from_pretrained('t5-small'),
        }
        print(f"已加载 {len(self.tokenizers)} 个 tokenizer")
    
    def compare(self, text):
        """比较不同 tokenizer 的编码结果"""
        results = {}
        
        for name, tokenizer in self.tokenizers.items():
            # 编码
            encoded = tokenizer.encode(text)
            
            # 解码
            decoded = tokenizer.decode(encoded)
            
            # 统计
            vocab_size = tokenizer.vocab_size
            seq_len = len(encoded)
            
            results[name] = {
                'encoded': encoded,
                'decoded': decoded,
                'vocab_size': vocab_size,
                'seq_len': seq_len,
                'compression': len(text) / seq_len if seq_len > 0 else 0
            }
        
        return results
    
    def print_comparison(self, text):
        """打印比较结果"""
        results = self.compare(text)
        
        print(f"\n文本: {text}")
        print(f"原始长度: {len(text)} 字符")
        print("="*60)
        
        for name, data in results.items():
            print(f"\n{name}:")
            print(f"  词汇表大小: {data['vocab_size']:,}")
            print(f"  序列长度: {data['seq_len']}")
            print(f"  压缩比: {data['compression']:.2f}x")
            print(f"  编码结果: {data['encoded'][:10]}..." if len(data['encoded']) > 10 else f"  编码结果: {data['encoded']}")
            print(f"  解码结果: {data['decoded'][:50]}..." if len(data['decoded']) > 50 else f"  解码结果: {data['decoded']}")
        
        # 找出最高效的
        best = min(results.items(), key=lambda x: x[1]['seq_len'])
        print(f"\n最高效: {best[0]} ({best[1]['seq_len']} tokens)")

# 使用示例
if __name__ == "__main__":
    comparator = TokenizerComparator()
    comparator.load_tokenizers()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "机器学习是人工智能的一个重要分支",
        "Machine learning is a subset of artificial intelligence"
    ]
    
    for text in test_texts:
        comparator.print_comparison(text)
```

## 练习题

### 基础题
1. **算法理解**：用自己的话解释 BPE 算法的工作原理和步骤。
2. **代码分析**：分析上述 BPE 实现中 `_merge_vocab` 函数的作用。

### 进阶题
3. **代码实现**：修改 BPE 实现，添加对中文的支持（处理中文字符）。
4. **性能优化**：优化 BPE 训练过程，使用更高效的数据结构。

### 思考题
5. **架构设计**：如果要设计一个支持 100 种语言的 Tokenizer，需要考虑哪些问题？

### GitHub 热门资源
1. **LLMs-from-scratch**
   - 相关章节：第 2 章《Tokenizer 实现》
2. **Happy-LLM**
   - 相关章节：实践部分《Tokenizer 实现》
3. **SentencePiece**
   - Google 的 SentencePiece 实现

### 重要论文
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015) - BPE 原始论文
- "Google's Neural Machine Translation System" (Wu et al., 2016) - WordPiece
- "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)

### 工具与库
- Hugging Face Tokenizers - 高性能 tokenizer 库
- SentencePiece - Google 的 tokenizer
- tiktoken - OpenAI 的 BPE tokenizer

### 学习资源
- Byte Pair Encoding Explained - BPE 图解
- The Illustrated Word2vec - 子词可视化
- Hugging Face Tokenizers Course - Tokenizer 教程

---

## Tokenizer 训练工作流

### 训练自定义 Tokenizer

```python
"""
Tokenizer 训练工作流
支持 BPE、WordPiece、Unigram 三种训练方式
"""
import os
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Tokenizer 配置"""
    # 训练数据
    corpus_files: List[str]
    output_dir: str = "./tokenizer"
    
    # 模型配置
    model_type: str = "BPE"  # BPE, WordPiece, Unigram
    vocab_size: int = 32000
    min_frequency: int = 2
    
    # 特殊 token
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<s>", "</s>", "<mask>"
    ])
    
    # 预分词
    pre_tokenizer: str = "byte_level"  # byte_level, whitespace, None


class TokenizerTrainer:
    """Tokenizer 训练器"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def train_bpe(self):
        """训练 BPE Tokenizer"""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        
        # 创建 tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # 设置预分词器
        if self.config.pre_tokenizer == "byte_level":
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 创建训练器
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # 训练
        logger.info(f"开始训练 BPE Tokenizer，词表大小: {self.config.vocab_size}")
        tokenizer.train(self.config.corpus_files, trainer)
        
        return tokenizer
    
    def train_wordpiece(self):
        """训练 WordPiece Tokenizer"""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        
        # 创建 tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
        
        # 设置预分词器
        if self.config.pre_tokenizer == "whitespace":
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # 创建训练器
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # 训练
        logger.info(f"开始训练 WordPiece Tokenizer，词表大小: {self.config.vocab_size}")
        tokenizer.train(self.config.corpus_files, trainer)
        
        return tokenizer
    
    def train_unigram(self):
        """训练 Unigram Tokenizer"""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        
        # 创建 tokenizer
        tokenizer = Tokenizer(models.Unigram())
        
        # 创建训练器
        trainer = trainers.UnigramTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # 训练
        logger.info(f"开始训练 Unigram Tokenizer，词表大小: {self.config.vocab_size}")
        tokenizer.train(self.config.corpus_files, trainer)
        
        return tokenizer
    
    def train(self) -> str:
        """执行训练"""
        # 选择训练方式
        if self.config.model_type == "BPE":
            tokenizer = self.train_bpe()
        elif self.config.model_type == "WordPiece":
            tokenizer = self.train_wordpiece()
        elif self.config.model_type == "Unigram":
            tokenizer = self.train_unigram()
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
        
        # 保存 tokenizer
        output_path = Path(self.config.output_dir) / "tokenizer.json"
        tokenizer.save(str(output_path))
        logger.info(f"Tokenizer 保存到: {output_path}")
        
        # 保存配置
        config_path = Path(self.config.output_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_type": self.config.model_type,
                "vocab_size": self.config.vocab_size,
                "special_tokens": self.config.special_tokens
            }, f, indent=2)
        
        # 保存词表
        vocab_path = Path(self.config.output_dir) / "vocab.json"
        vocab = tokenizer.get_vocab()
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        logger.info(f"词表大小: {len(vocab)}")
        return str(output_path)
    
    def evaluate(self, tokenizer, test_texts: List[str]):
        """评估 Tokenizer"""
        logger.info("评估 Tokenizer...")
        
        total_tokens = 0
        total_chars = 0
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            total_tokens += len(encoded.ids)
            total_chars += len(text)
        
        # 计算压缩率
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        
        logger.info(f"测试文本数: {len(test_texts)}")
        logger.info(f"总字符数: {total_chars}")
        logger.info(f"总 Token 数: {total_tokens}")
        logger.info(f"压缩率: {compression_ratio:.2f} chars/token")
        
        return {
            "num_texts": len(test_texts),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "compression_ratio": compression_ratio
        }


# 使用示例
if __name__ == "__main__":
    # 配置
    config = TokenizerConfig(
        corpus_files=["./data/train.txt"],
        output_dir="./tokenizer",
        model_type="BPE",
        vocab_size=32000,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
    )
    
    # 训练
    trainer = TokenizerTrainer(config)
    output_path = trainer.train()
    
    # 加载并测试
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(output_path)
    
    # 测试编码
    test_text = "大语言模型是人工智能的重要应用。"
    encoded = tokenizer.encode(test_text)
    print(f"原文: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    print(f"解码: {tokenizer.decode(encoded.ids)}")
```

### Tokenizer 评估脚本

```python
"""
Tokenizer 评估脚本
用于比较不同 Tokenizer 的性能
"""
from tokenizers import Tokenizer
from typing import List, Dict
import json


def evaluate_tokenizer(tokenizer_path: str, test_file: str) -> Dict:
    """评估单个 Tokenizer"""
    # 加载 tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 加载测试数据
    with open(test_file, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # 计算指标
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t).ids) for t in texts)
    
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "num_texts": len(texts),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "compression_ratio": total_chars / total_tokens,
        "avg_tokens_per_text": total_tokens / len(texts)
    }


def compare_tokenizers(tokenizer_paths: List[str], test_file: str):
    """比较多个 Tokenizer"""
    results = []
    
    for path in tokenizer_paths:
        metrics = evaluate_tokenizer(path, test_file)
        metrics["path"] = path
        results.append(metrics)
    
    # 打印比较结果
    print("\n" + "=" * 60)
    print("Tokenizer 比较结果")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['path']}:")
        print(f"  词表大小: {r['vocab_size']:,}")
        print(f"  压缩率: {r['compression_ratio']:.2f} chars/token")
        print(f"  平均 Token 数: {r['avg_tokens_per_text']:.1f}")
    
    return results


# 使用示例
if __name__ == "__main__":
    tokenizers = [
        "./tokenizer-bpe/tokenizer.json",
        "./tokenizer-wordpiece/tokenizer.json",
    ]
    
    compare_tokenizers(tokenizers, "./data/test.txt")
```

---

**更新日期：** 2026-03-30
