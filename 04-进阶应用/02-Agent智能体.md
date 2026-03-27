# Agent 智能体

## 学习目标
学完本节后，你将能够：
- 理解 Agent 智能体的核心概念和架构
- 掌握构建 Agent 系统的关键技术
- 了解不同类型的 Agent 和应用场景
- 能够构建简单的 Agent 应用

## 核心知识点

### 1. Agent 基础

#### 1.1 什么是 Agent
Agent（智能体）是一种能够感知环境、进行推理决策并采取行动以实现特定目标的系统。

**核心能力**：
- **感知（Perception）**：理解输入和环境状态
- **规划（Planning）**：制定实现目标的计划
- **行动（Action）**：执行计划中的操作
- **反思（Reflection）**：评估结果并调整策略

#### 1.2 Agent 架构
```
环境 → 感知模块 → 推理模块 → 决策模块 → 行动模块 → 环境
                                    ↑
                               记忆/知识库
```

#### 1.3 Agent vs 传统程序
| 维度 | 传统程序 | Agent |
|------|----------|-------|
| **目标** | 执行预定义任务 | 自主实现目标 |
| **灵活性** | 固定流程 | 动态调整 |
| **环境交互** | 被动接受输入 | 主动感知环境 |
| **学习能力** | 无 | 有（可选） |

### 2. Agent 类型

#### 2.1 基于 LLM 的 Agent
- **核心**：使用 LLM 作为推理引擎
- **代表**：AutoGPT、BabyAGI、LangChain Agent
- **优势**：强大的语言理解和生成能力

#### 2.2 反应式 Agent
- **特点**：直接对环境刺激做出反应
- **架构**：简单的条件-行动规则
- **适用**：简单、确定性的任务

#### 2.3 认知 Agent
- **特点**：具有内部世界模型和推理能力
- **架构**：BDI（信念-愿望-意图）模型
- **适用**：复杂、动态的环境

#### 2.4 多 Agent 系统
- **特点**：多个 Agent 协作完成任务
- **通信**：消息传递、共享记忆
- **协调**：协商、合作、竞争

### 3. Agent 关键技术

#### 3.1 记忆系统
- **短期记忆**：当前对话/任务上下文
- **长期记忆**：历史经验、知识库
- **记忆检索**：基于相似度的记忆召回
- **记忆更新**：经验的学习和存储

#### 3.2 规划技术
- **任务分解**：将复杂目标分解为子任务
- **计划生成**：制定实现目标的步骤序列
- **计划执行**：按步骤执行并监控进度
- **计划调整**：根据反馈调整计划

#### 3.3 工具使用
- **工具定义**：描述工具的功能和参数
- **工具选择**：根据任务选择合适的工具
- **工具调用**：执行工具并获取结果
- **结果整合**：将工具结果融入推理过程

#### 3.4 反思机制
- **自我评估**：评估当前状态和进展
- **错误检测**：识别执行中的错误
- **策略调整**：根据评估调整策略
- **经验总结**：从成功和失败中学习

### 4. Agent 开发框架

#### 4.1 LangChain Agent
- **核心概念**：工具、链、记忆
- **优势**：丰富的工具集成，灵活的链式调用
- **适用**：快速构建原型

#### 4.2 AutoGPT
- **特点**：自主目标分解和执行
- **优势**：高度自主，适合探索性任务
- **限制**：可能陷入循环，需要人工监督

#### 4.3 MetaGPT
- **特点**：多角色协作，软件开发流程
- **优势**：结构化的协作模式
- **适用**：复杂项目开发

## 代码示例

### 简单 Agent 实现

```python
from typing import List, Dict, Any, Callable
import json
import re

class Agent:
    """简单的 LLM Agent"""
    
    def __init__(self, name: str, llm_function: Callable):
        self.name = name
        self.llm = llm_function  # LLM 调用函数
        self.memory = []
        self.tools = {}
        self.goals = []
    
    def add_tool(self, name: str, description: str, func: Callable):
        """添加工具"""
        self.tools[name] = {
            'name': name,
            'description': description,
            'function': func
        }
    
    def set_goals(self, goals: List[str]):
        """设置目标"""
        self.goals = goals
    
    def perceive(self, input_text: str) -> Dict:
        """感知输入"""
        return {
            'input': input_text,
            'timestamp': 'now',
            'context': self.memory[-5:] if self.memory else []
        }
    
    def plan(self, perception: Dict) -> List[Dict]:
        """规划行动"""
        # 构建规划提示
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools.values()])
        goals_desc = "\n".join([f"- {g}" for g in self.goals])
        
        prompt = f"""你是一个智能体，需要完成以下目标：
{goals_desc}

可用工具：
{tools_desc}

当前输入：{perception['input']}
历史上下文：{json.dumps(perception['context'], ensure_ascii=False)}

请制定一个行动计划，使用以下格式：
思考：<你的思考过程>
行动：<工具名> <参数>
观察：<行动后的观察>

每次只执行一步。"""
        
        # 调用 LLM 获取计划
        response = self.llm(prompt)
        
        # 解析响应
        plan = self.parse_response(response)
        return plan
    
    def parse_response(self, response: str) -> List[Dict]:
        """解析 LLM 响应"""
        actions = []
        
        # 提取思考、行动、观察
        thought_match = re.search(r'思考[：:]\s*(.+?)(?=\n|$)', response)
        action_match = re.search(r'行动[：:]\s*(.+?)(?=\n|$)', response)
        observation_match = re.search(r'观察[：:]\s*(.+?)(?=\n|$)', response)
        
        if action_match:
            action_text = action_match.group(1).strip()
            # 解析工具和参数
            parts = action_text.split(' ', 1)
            tool_name = parts[0] if parts else ''
            params = parts[1] if len(parts) > 1 else ''
            
            actions.append({
                'type': 'tool',
                'tool': tool_name,
                'params': params,
                'thought': thought_match.group(1).strip() if thought_match else '',
                'expected_observation': observation_match.group(1).strip() if observation_match else ''
            })
        
        return actions
    
    def execute(self, action: Dict) -> str:
        """执行行动"""
        if action['type'] == 'tool':
            tool_name = action['tool']
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    # 调用工具函数
                    result = tool['function'](action['params'])
                    return f"工具 {tool_name} 执行成功：{result}"
                except Exception as e:
                    return f"工具 {tool_name} 执行失败：{str(e)}"
            else:
                return f"未知工具：{tool_name}"
        else:
            return f"未知行动类型：{action['type']}"
    
    def reflect(self, observation: str) -> str:
        """反思"""
        # 简单反思：记录观察
        self.memory.append({
            'timestamp': 'now',
            'observation': observation
        })
        
        # 限制记忆长度
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]
        
        return f"已记录观察：{observation[:50]}..."
    
    def run(self, input_text: str, max_steps: int = 5) -> Dict:
        """运行 Agent"""
        print(f"[{self.name}] 开始处理：{input_text}")
        
        results = {
            'input': input_text,
            'steps': [],
            'final_output': ''
        }
        
        current_input = input_text
        
        for step in range(max_steps):
            print(f"\n--- 步骤 {step + 1} ---")
            
            # 1. 感知
            perception = self.perceive(current_input)
            print(f"感知：{perception['input']}")
            
            # 2. 规划
            plan = self.plan(perception)
            if not plan:
                print("无法生成计划")
                break
            
            action = plan[0]
            print(f"计划：{action['thought']}")
            print(f"行动：{action['tool']} {action['params']}")
            
            # 3. 执行
            observation = self.execute(action)
            print(f"观察：{observation}")
            
            # 4. 反思
            reflection = self.reflect(observation)
            print(f"反思：{reflection}")
            
            # 记录步骤
            results['steps'].append({
                'step': step + 1,
                'action': action,
                'observation': observation,
                'reflection': reflection
            })
            
            # 更新输入
            current_input = observation
            
            # 检查是否完成
            if "完成" in observation or "成功" in observation:
                results['final_output'] = observation
                break
        
        print(f"\n[{self.name}] 处理完成")
        return results

# 示例工具函数
def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        # 安全计算（仅允许基本运算）
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        else:
            return "错误：包含不允许的字符"
    except Exception as e:
        return f"计算错误：{str(e)}"

def web_search(query: str) -> str:
    """模拟搜索工具"""
    # 实际实现中应调用搜索 API
    return f"搜索 '{query}' 的结果：找到相关信息（模拟）"

def python_exec(code: str) -> str:
    """Python 执行工具（简化版）"""
    try:
        # 注意：实际使用时需要安全沙箱
        local_vars = {}
        exec(f"result = {code}", {}, local_vars)
        return f"执行结果：{local_vars.get('result', '无结果')}"
    except Exception as e:
        return f"执行错误：{str(e)}"

# 使用示例
if __name__ == "__main__":
    # 模拟 LLM 函数（实际应调用真实 API）
    def mock_llm(prompt: str) -> str:
        # 简单模拟响应
        if "计算" in prompt:
            return "思考：用户需要计算\n行动：calculator 2+3*4\n观察：等待计算结果"
        elif "搜索" in prompt:
            return "思考：用户需要搜索信息\n行动：web_search 人工智能发展\n观察：等待搜索结果"
        else:
            return "思考：需要更多信息\n行动：\n观察：等待进一步输入"
    
    # 创建 Agent
    agent = Agent(name="Assistant", llm_function=mock_llm)
    
    # 添加工具
    agent.add_tool("calculator", "执行数学计算", calculator)
    agent.add_tool("web_search", "搜索网络信息", web_search)
    agent.add_tool("python_exec", "执行 Python 代码", python_exec)
    
    # 设置目标
    agent.set_goals(["帮助用户完成任务", "提供准确信息"])
    
    # 运行 Agent
    result = agent.run("计算 2+3*4 的结果", max_steps=3)
    
    print("\n" + "="*50)
    print("最终结果：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

### 多 Agent 协作系统

```python
from typing import List, Dict
import asyncio

class MultiAgentSystem:
    """多 Agent 协作系统"""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = []
        self.shared_memory = {}
    
    def add_agent(self, agent_id: str, agent):
        """添加 Agent"""
        self.agents[agent_id] = agent
        agent.system = self  # 让 Agent 能访问系统
    
    def send_message(self, from_id: str, to_id: str, message: Dict):
        """发送消息"""
        self.message_queue.append({
            'from': from_id,
            'to': to_id,
            'message': message,
            'timestamp': 'now'
        })
    
    def broadcast(self, from_id: str, message: Dict):
        """广播消息"""
        for agent_id in self.agents:
            if agent_id != from_id:
                self.send_message(from_id, agent_id, message)
    
    def get_messages(self, agent_id: str) -> List[Dict]:
        """获取 Agent 的消息"""
        messages = [msg for msg in self.message_queue if msg['to'] == agent_id]
        # 移除已读消息
        self.message_queue = [msg for msg in self.message_queue if msg['to'] != agent_id]
        return messages
    
    def update_shared_memory(self, key: str, value: Any):
        """更新共享记忆"""
        self.shared_memory[key] = value
    
    def get_shared_memory(self, key: str) -> Any:
        """获取共享记忆"""
        return self.shared_memory.get(key)

class CollaborativeAgent:
    """协作 Agent"""
    
    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.system = None
    
    def process_message(self, message: Dict) -> Dict:
        """处理消息"""
        # 基于角色的处理逻辑
        if self.role == "researcher":
            return self._research_task(message)
        elif self.role == "writer":
            return self._writing_task(message)
        elif self.role == "reviewer":
            return self._review_task(message)
        else:
            return {"error": "未知角色"}
    
    def _research_task(self, message: Dict) -> Dict:
        """研究任务"""
        topic = message.get('topic', '')
        # 模拟研究
        research_result = f"关于 {topic} 的研究结果：这是模拟的研究数据。"
        
        # 发送给 writer
        self.system.send_message(
            self.agent_id,
            "writer",
            {"type": "research_result", "topic": topic, "content": research_result}
        )
        
        return {"status": "completed", "result": research_result}
    
    def _writing_task(self, message: Dict) -> Dict:
        """写作任务"""
        content = message.get('content', '')
        # 模拟写作
        article = f"根据研究：{content}\n\n这是生成的文章内容。"
        
        # 发送给 reviewer
        self.system.send_message(
            self.agent_id,
            "reviewer",
            {"type": "draft", "content": article}
        )
        
        return {"status": "completed", "article": article}
    
    def _review_task(self, message: Dict) -> Dict:
        """审核任务"""
        content = message.get('content', '')
        # 模拟审核
        feedback = f"审核意见：文章内容良好，建议：{content}。"
        
        # 更新共享记忆
        self.system.update_shared_memory("final_article", content)
        
        return {"status": "completed", "feedback": feedback}

# 使用示例
if __name__ == "__main__":
    # 创建多 Agent 系统
    system = MultiAgentSystem()
    
    # 创建协作 Agent
    researcher = CollaborativeAgent("researcher", "researcher")
    writer = CollaborativeAgent("writer", "writer")
    reviewer = CollaborativeAgent("reviewer", "reviewer")
    
    # 添加到系统
    system.add_agent("researcher", researcher)
    system.add_agent("writer", writer)
    system.add_agent("reviewer", reviewer)
    
    # 模拟协作流程
    print("=== 多 Agent 协作演示 ===")
    
    # 1. 研究员开始研究
    researcher_result = researcher.process_message({
        "type": "research_request",
        "topic": "人工智能发展趋势"
    })
    print(f"研究员结果: {researcher_result}")
    
    # 2. 检查消息队列
    writer_messages = system.get_messages("writer")
    print(f"\nWriter 收到 {len(writer_messages)} 条消息")
    
    # 3. Writer 写作
    if writer_messages:
        writer_result = writer.process_message(writer_messages[0]['message'])
        print(f"Writer 结果: {writer_result}")
    
    # 4. 检查消息队列
    reviewer_messages = system.get_messages("reviewer")
    print(f"\nReviewer 收到 {len(reviewer_messages)} 条消息")
    
    # 5. Reviewer 审核
    if reviewer_messages:
        reviewer_result = reviewer.process_message(reviewer_messages[0]['message'])
        print(f"Reviewer 结果: {reviewer_result}")
    
    # 6. 查看共享记忆
    final_article = system.get_shared_memory("final_article")
    print(f"\n最终文章: {final_article}")
```

## Agent 配置与部署

### 1. Agent 配置模板

**基础配置文件：** `agent_config.yaml`

```yaml
agent:
  name: "智能助手"
  version: "1.0.0"
  description: "基于 LLM 的智能 Agent"

# LLM 配置
llm:
  provider: "openai"  # openai, anthropic, local
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  api_key: "${OPENAI_API_KEY}"

# 记忆配置
memory:
  type: "hybrid"  # short_term, long_term, hybrid
  short_term:
    max_tokens: 4000
    window_size: 10  # 保留最近10轮对话
  long_term:
    storage: "chroma"  # chroma, pinecone, local
    collection: "agent_memory"
    embedding_model: "text-embedding-ada-002"

# 工具配置
tools:
  enabled: true
  list:
    - name: "web_search"
      description: "搜索互联网获取信息"
      parameters:
        query: "string"
    
    - name: "calculator"
      description: "执行数学计算"
      parameters:
        expression: "string"
    
    - name: "file_manager"
      description: "管理本地文件"
      parameters:
        action: "string"  # read, write, list
        path: "string"

# 规划配置
planning:
  strategy: "react"  # react, cot, tot
  max_steps: 10
  reflection_enabled: true

# 安全配置
safety:
  content_filter: true
  max_tool_calls_per_minute: 30
  allowed_domains: ["google.com", "wikipedia.org"]
  blocked_actions: ["delete_file", "send_email"]

# 日志配置
logging:
  level: "info"
  file: "./logs/agent.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 2. 部署配置

**Docker 部署：** `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建日志目录
RUN mkdir -p logs

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV AGENT_ENV=production

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "agent_server.py"]
```

**Docker Compose：** `docker-compose.yaml`

```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AGENT_ENV=production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - redis
      - chroma
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  redis_data:
  chroma_data:
```

### 3. 环境变量配置

**.env 文件：**

```bash
# LLM 配置
OPENAI_API_KEY=sk-xxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx

# 数据库配置
REDIS_URL=redis://localhost:6379
CHROMA_URL=http://localhost:8001

# Agent 配置
AGENT_NAME=my-agent
AGENT_ENV=development
LOG_LEVEL=debug

# 安全配置
ALLOWED_DOMAINS=google.com,wikipedia.org,github.com
MAX_REQUESTS_PER_MINUTE=60
```

### 4. 监控与告警

**Prometheus 配置：**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'chroma'
    static_configs:
      - targets: ['localhost:8001']
```

**Grafana 仪表盘指标：**

- Agent 响应时间
- 工具调用成功率
- Token 使用量
- 错误率
- 内存使用情况

---

## 练习题

### 基础题
1. **概念理解**：解释 Agent 的核心组件（感知、规划、行动、反思）及其作用。
2. **架构分析**：比较基于 LLM 的 Agent 和传统 Agent 的区别。

### 进阶题
3. **系统设计**：设计一个多 Agent 协作系统，用于完成软件开发任务。
4. **工具集成**：为 Agent 添加至少 3 个不同的工具，并设计工具选择策略。

### 思考题
5. **未来方向**：Agent 技术未来可能的发展方向是什么？如何实现真正的自主智能体？

### GitHub 热门资源
1. **AutoGPT** - 自主 Agent
2. **BabyAGI** - 简单 Agent
3. **LangChain** - Agent 框架
4. **MetaGPT** - 多 Agent 协作

### 重要论文
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
- "Generative Agents: Interactive Simulacra of Human Behavior" (Park et al., 2023)
- "CAMEL: Communicative Agents for "Mind" Exploration" (Li et al., 2023)

### 工具与框架
- LangChain Agents - Agent 开发
- CrewAI - 多 Agent 协作
- AutoGen - 多 Agent 对话

### 学习资源
- LangChain Agent Tutorial
- AutoGPT Guide
- Agent Design Patterns - Agent 设计模式