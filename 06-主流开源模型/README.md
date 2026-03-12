# 主流开源模型

> 本章节详细介绍当前主流开源大语言模型的架构、特点、性能及部署方法，涵盖 Llama、Qwen、Mistral、DeepSeek、GLM 等热门模型系列。

## 模型列表

| 模型系列 | 机构 | 代表模型 | 特点 |
|---------|------|---------|------|
| [Llama 系列](./01-Llama 系列详解.md) | Meta | Llama3-70B | 开源标杆，生态完善 |
| [Qwen 系列](./02-Qwen 系列详解.md) | 阿里 | Qwen2.5-72B | 中文优化，MoE 架构 |
| [Mistral 系列](./03-Mistral 系列详解.md) | Mistral AI | Mixtral 8x22B | 高效 MoE，性能优异 |
| [DeepSeek 系列](./04-DeepSeek 系列详解.md) | 深度求索 | DeepSeek-V3 | MLA 注意力，高性价比 |
| [GLM 系列](./05-GLM 系列详解.md) | 智谱 AI | GLM-4 | 混合架构，中文领先 |
| [其他国产模型](./06-其他国产模型.md) | 多家 | Yi/InternLM/Baichuan | 多样化选择 |

## 快速选择指南

### 按参数量选择

| 规模 | 推荐模型 | 显存需求 (FP16) | 适用场景 |
|------|---------|---------------|---------|
| 小模型 (<10B) | Llama3-8B, Qwen2.5-7B, Mistral 7B | 14-16 GB | 本地部署、边缘设备 |
| 中模型 (10-50B) | Qwen2.5-32B, Yi-34B, Mixtral 8x7B | 60-80 GB | 企业应用、专业任务 |
| 大模型 (>50B) | Llama3-70B, Qwen2.5-72B, DeepSeek-V2 | 140-280 GB | 云端服务、高性能需求 |

### 按任务选择

| 任务 | 推荐模型 | 理由 |
|------|---------|------|
| 通用对话 | Llama3-8B, Qwen2.5-7B | 平衡性能与资源 |
| 中文任务 | Qwen2.5, GLM-4, Baichuan2 | 中文优化 |
| 代码生成 | DeepSeek-Coder, Qwen2.5-Coder | 代码能力强化 |
| 长文档 | Qwen2.5-128K, Yi-200K, GLM-4 | 长上下文支持 |
| 多模态 | Qwen-VL, GLM-4V | 图像理解 |
| 工具调用 | Qwen2.5, ChatGLM3, InternLM2 | 原生 Function Calling |

### 按硬件选择

| 硬件配置 | 推荐模型 | 量化建议 |
|---------|---------|---------|
| RTX 3060 12G | Llama3-8B, Qwen2.5-7B, Mistral 7B | INT4 可运行 13B |
| RTX 3090 24G | Llama3-70B(INT4), Qwen2.5-32B | INT4 推荐 |
| RTX 4090 24G | 同上，推理更快 | INT4/FP16 |
| 2×A100 80G | Llama3-70B, Qwen2.5-72B | FP16/INT8 |
| 4×A100 80G | DeepSeek-V2, Mixtral 8x22B | FP16 |

## 性能对比总览

### 综合基准

| 模型 | MMLU | GSM8K | HumanEval | MATH | 上下文 |
|------|------|-------|-----------|------|--------|
| Llama3-70B | 79.5 | 84.2 | 72.8 | 48.7 | 8K |
| Llama3.1-405B | 88.6 | 92.1 | 89.0 | 65.2 | 128K |
| Qwen2.5-72B | 82.3 | 89.1 | 78.5 | 56.2 | 128K |
| Mixtral 8x22B | 78.2 | 81.5 | 70.3 | 45.3 | 64K |
| DeepSeek-V2.5 | 80.5 | 85.3 | 75.8 | 52.1 | 128K |
| GLM-4 | 82.5 | 87.2 | 76.5 | 54.8 | 128K |
| Yi-34B | 75.9 | 73.5 | 58.5 | 42.3 | 200K |

### 推理速度对比（A100 80G，tokens/s）

| 模型 | FP16 | INT8 | INT4 |
|------|------|------|------|
| Llama3-8B | 120 | 180 | 250 |
| Llama3-70B | 25 | 45 | 80 |
| Qwen2.5-7B | 115 | 175 | 240 |
| Qwen2.5-72B | 22 | 40 | 75 |
| Mistral 7B | 130 | 190 | 260 |
| Mixtral 8x7B | 100 | 150 | 200 |
| DeepSeek-V2 | 85 | 130 | 180 |

## 部署方式对比

### Ollama（推荐新手）

```bash
# 安装
curl -fsSL https://ollama.ai/install.sh | sh

# 运行模型
ollama run llama3:8b
ollama run qwen2.5:7b
ollama run mistral:7b
ollama run deepseek-coder:6.7b
```

**优点：** 一键安装，自动下载，简单易用  
**缺点：** 自定义选项较少

### vLLM（推荐生产）

```bash
pip install vllm

python -m vllm.entrypoints.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --max-model-len 8192
```

**优点：** 高吞吐，PagedAttention，生产级  
**缺点：** 配置较复杂

### llama.cpp（推荐本地）

```bash
# 编译
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 量化
python convert-hf-to-gguf.py model
./quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# 运行
./main -m model-q4_k_m.gguf -p "Hello" -n 256
```

**优点：** CPU 可运行，量化效果好，跨平台  
**缺点：** 需要转换模型格式

## 模型选择决策树

```
开始
│
├─ 需要本地运行？
│  ├─ 是 → 显存多少？
│  │       ├─ <8GB → Yi-6B, Qwen2.5-3B (INT4)
│  │       ├─ 8-16GB → Llama3-8B, Qwen2.5-7B, Mistral 7B
│  │       └─ 16-24GB → Llama3-70B(INT4), Qwen2.5-32B(INT4)
│  │
│  └─ 否 → 继续
│
├─ 主要用途？
│  ├─ 中文任务 → Qwen2.5, GLM-4, Baichuan2
│  ├─ 代码生成 → DeepSeek-Coder, Qwen2.5-Coder
│  ├─ 长文档 → Yi-34B-200K, Qwen2.5-128K
│  └─ 通用 → Llama3, Mixtral
│
├─ 性能要求？
│  ├─ 最高 → Llama3.1-405B, DeepSeek-V3, Qwen2.5-72B
│  ├─ 平衡 → Llama3-70B, Qwen2.5-32B, Mixtral 8x22B
│  └─ 轻量 → Llama3-8B, Qwen2.5-7B, Mistral 7B
│
└─ 预算限制？
   ├─ 免费开源 → 全部上述模型
   ├─ 可接受 API → GPT-4, Claude, Gemini
   └─ 企业采购 → 联系模型厂商获取商用授权
```

## 学习建议

### 入门路线

1. **第一阶段（1-2 周）**
   - 了解 LLM 基础概念
   - 使用 Ollama 运行 Llama3-8B
   - 尝试简单对话和问答

2. **第二阶段（2-4 周）**
   - 学习 Prompt Engineering
   - 尝试不同模型对比
   - 了解量化和部署

3. **第三阶段（1-2 月）**
   - 学习 RAG 技术
   - 尝试模型微调（LoRA）
   - 构建实际应用

4. **第四阶段（持续）**
   - 关注新模型发布
   - 参与开源项目
   - 探索前沿技术

## 更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-03-12 | 初始版本，包含 6 个模型系列详解 |

## 相关章节

- [02-核心原理](../02-核心原理/) — Transformer 架构、注意力机制
- [04-进阶应用](../04-进阶应用/) — RAG、Agent、模型评测
- [05-实战项目](../05-实战项目/) — 对话机器人、微调专属模型

---

**维护者：** OpenClaw  
**更新日期：** 2026-03-12
