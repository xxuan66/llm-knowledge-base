# 实战项目三：多 Agent 协作系统

> 构建多个 AI Agent 协同工作的系统，模拟真实团队的协作流程，完成复杂任务。

## 项目目标

设计并实现一个多 Agent 系统，包含：

1. **角色分工** — 不同 Agent 负责不同任务
2. **任务编排** — 自动分解和分配任务
3. **通信机制** — Agent 之间的消息传递
4. **协作协议** — 定义协作规则和流程
5. **质量控制** — 审核和反馈机制

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator (调度器)                   │
│                   任务分解 / 流程控制                      │
└──────────┬──────────────────────────────────┬───────────┘
           │                                  │
    ┌──────▼──────┐                    ┌──────▼──────┐
    │  研究员 Agent │                    │  规划师 Agent │
    │  (Research)  │                    │  (Planner)   │
    └──────┬──────┘                    └──────┬──────┘
           │                                  │
    ┌──────▼──────┐                    ┌──────▼──────┐
    │  开发者 Agent │                    │  测试者 Agent │
    │  (Developer) │  ───────────────► │  (Tester)    │
    └──────┬──────┘                    └──────┬──────┘
           │                                  │
           └──────────────┬───────────────────┘
                   ┌──────▼──────┐
                   │  审核者 Agent │
                   │  (Reviewer)  │
                   └─────────────┘
```

## 角色定义

### 1. Orchestrator（调度器）

负责整体流程控制和任务分解：

```python
class Orchestrator:
    def __init__(self, agents: dict):
        self.agents = agents
        self.task_queue = []
        self.results = {}
    
    def decompose_task(self, user_request: str) -> list[dict]:
        """将用户请求分解为子任务"""
        prompt = f"""将以下任务分解为子任务：
{user_request}

请以 JSON 格式输出：
[{{"task": "任务描述", "agent": "角色名", "dependencies": ["依赖任务ID"]}}]
"""
        # 调用 LLM 解析
        subtasks = self.llm_parse(prompt)
        return subtasks
    
    def execute(self, user_request: str):
        """执行完整流程"""
        # 1. 任务分解
        subtasks = self.decompose_task(user_request)
        
        # 2. 按依赖顺序执行
        completed = set()
        while len(completed) < len(subtasks):
            for task in subtasks:
                if task["id"] in completed:
                    continue
                if all(dep in completed for dep in task["dependencies"]):
                    # 执行任务
                    result = self.agents[task["agent"]].execute(task)
                    self.results[task["id"]] = result
                    completed.add(task["id"])
        
        return self.aggregate_results()
```

### 2. Researcher Agent（研究员）

负责信息收集和调研：

```python
class ResearcherAgent:
    def __init__(self, tools: list):
        self.tools = tools  # 搜索、网页抓取等
    
    def execute(self, task: dict) -> dict:
        """执行调研任务"""
        # 1. 制定调研计划
        plan = self.create_research_plan(task["task"])
        
        # 2. 执行调研
        findings = []
        for step in plan:
            results = self.search(step["query"])
            findings.extend(results)
        
        # 3. 整理报告
        report = self.summarize_findings(findings)
        
        return {
            "agent": "researcher",
            "task": task["task"],
            "report": report,
            "sources": findings
        }
    
    def search(self, query: str) -> list:
        """调用搜索工具"""
        # 可以使用 web_search、文档检索等
        return self.tools["web_search"](query)
```

### 3. Developer Agent（开发者）

负责代码编写和实现：

```python
class DeveloperAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def execute(self, task: dict, context: dict = None) -> dict:
        """执行开发任务"""
        prompt = f"""根据以下需求编写代码：

需求：{task['task']}

{f"参考信息：{context['report']}" if context else ""}

要求：
1. 代码结构清晰，有注释
2. 包含必要的错误处理
3. 提供使用示例
"""
        
        code = self.llm.generate(prompt)
        
        return {
            "agent": "developer",
            "task": task["task"],
            "code": code,
            "language": self.detect_language(code)
        }
```

### 4. Tester Agent（测试者）

负责测试和质量保证：

```python
class TesterAgent:
    def execute(self, task: dict, code: str) -> dict:
        """执行测试"""
        # 1. 分析代码
        analysis = self.analyze_code(code)
        
        # 2. 生成测试用例
        test_cases = self.generate_tests(code)
        
        # 3. 运行测试
        results = self.run_tests(code, test_cases)
        
        # 4. 生成报告
        return {
            "agent": "tester",
            "analysis": analysis,
            "test_cases": test_cases,
            "results": results,
            "passed": all(r["passed"] for r in results)
        }
```

### 5. Reviewer Agent（审核者）

负责最终审核和反馈：

```python
class ReviewerAgent:
    def execute(self, task: dict, all_results: dict) -> dict:
        """执行最终审核"""
        prompt = f"""请审核以下任务的完成情况：

任务：{task['task']}

研究员报告：{all_results.get('research', {}).get('report', 'N/A')}
开发者代码：{all_results.get('code', 'N/A')[:500]}
测试结果：{all_results.get('test', {}).get('results', 'N/A')}

请评估：
1. 是否满足原始需求？
2. 代码质量如何？
3. 测试覆盖是否充分？
4. 是否需要改进？

以 JSON 格式输出评分（1-10）和改进建议。"""
        
        review = self.llm.generate(prompt)
        
        return {
            "agent": "reviewer",
            "review": review,
            "approved": self.should_approve(review)
        }
```

## 通信机制

### 消息格式

```python
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: dict
    message_type: str  # "task", "result", "query", "feedback"
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = None  # 关联的任务ID
```

### 消息队列

```python
class MessageBus:
    def __init__(self):
        self.queues = {}  # agent_name -> queue
        self.message_log = []
    
    def send(self, message: AgentMessage):
        """发送消息"""
        self.queues.setdefault(message.receiver, []).append(message)
        self.message_log.append(message)
    
    def receive(self, agent_name: str) -> AgentMessage:
        """接收消息"""
        queue = self.queues.get(agent_name, [])
        if queue:
            return queue.pop(0)
        return None
    
    def broadcast(self, sender: str, content: dict, receivers: list[str]):
        """广播消息"""
        for receiver in receivers:
            self.send(AgentMessage(
                sender=sender,
                receiver=receiver,
                content=content,
                message_type="broadcast"
            ))
```

## 协作流程示例

### 场景：开发一个 Web 应用

```python
# 1. 用户输入
user_request = "开发一个 TODO List Web 应用，支持增删改查，使用 React + FastAPI"

# 2. Orchestrator 分解任务
tasks = orchestrator.decompose_task(user_request)
# 输出：
# [
#   {"id": "t1", "task": "调研 TODO List 应用的最佳实践", "agent": "researcher"},
#   {"id": "t2", "task": "设计数据库 Schema", "agent": "developer", "dependencies": ["t1"]},
#   {"id": "t3", "task": "编写 FastAPI 后端代码", "agent": "developer", "dependencies": ["t2"]},
#   {"id": "t4", "task": "编写 React 前端代码", "agent": "developer", "dependencies": ["t2"]},
#   {"id": "t5", "task": "编写测试用例", "agent": "tester", "dependencies": ["t3", "t4"]},
#   {"id": "t6", "task": "最终审核", "agent": "reviewer", "dependencies": ["t5"]}
# ]

# 3. 执行流程
results = orchestrator.execute(user_request)
```

## 使用框架

### 方案一：CrewAI

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert in web research",
    tools=[search_tool]
)

developer = Agent(
    role="Developer",
    goal="Write clean, working code",
    backstory="Senior software engineer",
    tools=[code_execution_tool]
)

task1 = Task(
    description="Research TODO app best practices",
    agent=researcher,
    expected_output="Research report"
)

task2 = Task(
    description="Build a TODO API with FastAPI",
    agent=developer,
    expected_output="Working Python code"
)

crew = Crew(
    agents=[researcher, developer],
    tasks=[task1, task2],
    verbose=True
)

result = crew.kickoff()
```

### 方案二：LangGraph

```python
from langgraph.graph import Graph

# 定义节点
def research_node(state):
    # 执行调研
    return {"research": "调研结果"}

def develop_node(state):
    # 执行开发
    return {"code": "开发结果"}

def review_node(state):
    # 执行审核
    return {"approved": True}

# 构建图
workflow = Graph()
workflow.add_node("research", research_node)
workflow.add_node("develop", develop_node)
workflow.add_node("review", review_node)

workflow.add_edge("research", "develop")
workflow.add_edge("develop", "review")

workflow.set_entry_point("research")
workflow.set_finish_point("review")

app = workflow.compile()
```

### 方案三：自定义实现（推荐学习）

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {
            "researcher": ResearcherAgent(),
            "developer": DeveloperAgent(),
            "tester": TesterAgent(),
            "reviewer": ReviewerAgent()
        }
        self.orchestrator = Orchestrator(self.agents)
        self.message_bus = MessageBus()
    
    def run(self, task: str):
        """执行任务"""
        return self.orchestrator.execute(task)
```

## 项目结构

```
multi-agent/
├── agents/
│   ├── __init__.py
│   ├── base.py           # Agent 基类
│   ├── researcher.py
│   ├── developer.py
│   ├── tester.py
│   └── reviewer.py
├── core/
│   ├── orchestrator.py   # 调度器
│   ├── message_bus.py    # 消息总线
│   └── task.py           # 任务定义
├── tools/
│   ├── search.py         # 搜索工具
│   ├── code_runner.py    # 代码执行
│   └── web_scraper.py    # 网页抓取
├── main.py
└── README.md
```

## 调试技巧

### 1. 可视化流程

```python
# 记录每个步骤的执行
def visualize_execution(results):
    for task_id, result in results.items():
        print(f"[{task_id}] {result['agent']}: {result['status']}")
```

### 2. 单步调试

```python
# 允许人工介入
class HumanInTheLoopOrchestrator(Orchestrator):
    def execute(self, task):
        for subtask in self.decompose_task(task):
            # 在每个步骤前暂停，等待人工确认
            if self.confirm(f"执行: {subtask['task']}?"):
                result = self.agents[subtask["agent"]].execute(subtask)
                self.results[subtask["id"]] = result
```

## 扩展阅读

- CrewAI 文档
- LangGraph 教程
- AutoGen 论文
- OpenClaw Agent 框架

---

**项目难度：** ⭐⭐⭐⭐ (高级)  
**预计用时：** 3-5 天  
**更新日期：** 2026-03-12
