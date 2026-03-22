# MCP 实际应用场景与最佳实践

本文介绍 MCP 在真实项目中的应用场景、设计模式和最佳实践。

---

## 应用场景总览

### 1. 企业数据集成

**场景：** 企业聊天机器人需要访问多个内部系统

```
┌─────────────────┐
│  企业聊天机器人  │
│   (MCP Host)    │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ 数据库  │ │ CRM    │ │ ERP    │ │ 文档库  │
│ Server │ │ Server │ │ Server │ │ Server │
└────────┘ └────────┘ └────────┘ └────────┘
```

**实现方案：**

```python
# 数据库 MCP 服务器示例
from mcp.server.fastmcp import FastMCP
import asyncpg

mcp = FastMCP("enterprise-database")

@mcp.tool()
async def query_sales_data(region: str, start_date: str, end_date: str) -> str:
    """查询销售数据。
    
    Args:
        region: 区域（如 "华东"、"华北"）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
    """
    conn = await asyncpg.connect("postgresql://...")
    
    query = """
        SELECT product, SUM(amount) as total
        FROM sales
        WHERE region = $1 AND date BETWEEN $2 AND $3
        GROUP BY product
        ORDER BY total DESC
        LIMIT 10
    """
    
    rows = await conn.fetch(query, region, start_date, end_date)
    await conn.close()
    
    return "\n".join([f"{row['product']}: {row['total']}" for row in rows])


@mcp.resource("db://schema/sales")
def get_sales_schema() -> str:
    """提供销售表 schema 作为资源。"""
    return """
    CREATE TABLE sales (
        id SERIAL PRIMARY KEY,
        product VARCHAR(100),
        region VARCHAR(50),
        amount DECIMAL(10,2),
        date DATE
    );
    """
```

**优势：**
- 统一的接口访问不同系统
- AI 可以理解数据结构并生成查询
- 安全控制（只暴露必要的工具）

---

### 2. 开发工具集成

**场景：** AI 编程助手需要访问 Git、测试、部署等工具

**GitHub MCP 服务器示例：**

```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("github-tools")

GITHUB_TOKEN = "ghp_..."  # 从环境变量读取

@mcp.tool()
async def create_issue(repo: str, title: str, body: str, labels: list[str] = None) -> str:
    """创建 GitHub Issue。
    
    Args:
        repo: 仓库全名（如 "owner/repo"）
        title: Issue 标题
        body: Issue 内容
        labels: 标签列表
    """
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {"title": title, "body": body}
    if labels:
        data["labels"] = labels
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
    
    return f"Issue 已创建：{result['html_url']}"


@mcp.tool()
async def get_pull_requests(repo: str, state: str = "open") -> str:
    """获取仓库的 Pull Requests。
    
    Args:
        repo: 仓库全名
        state: 状态（open/closed/all）
    """
    url = f"https://api.github.com/repos/{repo}/pulls?state={state}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        prs = response.json()
    
    result = []
    for pr in prs[:10]:  # 只显示前 10 个
        result.append(f"#{pr['number']} {pr['title']} by {pr['user']['login']}")
    
    return "\n".join(result)


@mcp.tool()
async def get_ci_status(repo: str, pr_number: int) -> str:
    """获取 CI/CD 构建状态。
    
    Args:
        repo: 仓库全名
        pr_number: PR 编号
    """
    # 获取 PR 的 HEAD SHA
    pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    
    async with httpx.AsyncClient() as client:
        pr_response = await client.get(
            pr_url,
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}
        )
        pr_response.raise_for_status()
        sha = pr_response.json()["head"]["sha"]
    
    # 获取检查状态
        checks_url = f"https://api.github.com/repos/{repo}/commits/{sha}/check-runs"
        checks_response = await client.get(
            checks_url,
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}
        )
        checks_response.raise_for_status()
        checks = checks_response.json()
    
    result = []
    for check in checks["check_runs"]:
        status = check["status"]
        conclusion = check.get("conclusion", "pending")
        result.append(f"{check['name']}: {status} - {conclusion}")
    
    return "\n".join(result)
```

**使用场景：**
- AI 助手帮助开发者创建 Issue
- 自动检查 PR 状态和 CI 结果
- 生成发布说明

---

### 3. 个人助手集成

**场景：** 个人 AI 助手访问日历、邮件、笔记等

**日历 MCP 服务器示例：**

```python
from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

mcp = FastMCP("personal-calendar")

@mcp.tool()
async def get_calendar_events(date: str, days: int = 7) -> str:
    """获取日历事件。
    
    Args:
        date: 开始日期（YYYY-MM-DD）
        days: 获取多少天的事件
    """
    creds = Credentials.from_authorized_user_file("token.json")
    service = build("calendar", "v3", credentials=creds)
    
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=days)
    
    events_result = service.events().list(
        calendarId="primary",
        timeMin=start_dt.isoformat() + "Z",
        timeMax=end_dt.isoformat() + "Z",
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    
    events = events_result.get("items", [])
    
    if not events:
        return "这段时间没有安排。"
    
    result = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        result.append(f"{start}: {event['summary']}")
    
    return "\n".join(result)


@mcp.tool()
async def create_calendar_event(summary: str, start_time: str, end_time: str, description: str = None) -> str:
    """创建日历事件。
    
    Args:
        summary: 事件标题
        start_time: 开始时间（ISO 8601 格式）
        end_time: 结束时间（ISO 8601 格式）
        description: 事件描述
    """
    creds = Credentials.from_authorized_user_file("token.json")
    service = build("calendar", "v3", credentials=creds)
    
    event = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": "Asia/Shanghai"},
        "end": {"dateTime": end_time, "timeZone": "Asia/Shanghai"},
    }
    
    if description:
        event["description"] = description
    
    event = service.events().insert(calendarId="primary", body=event).execute()
    return f"事件已创建：{event['htmlLink']}"
```

---

### 4. 跨平台自动化

**场景：** 连接不同平台实现自动化工作流

**Notion + Slack 集成示例：**

```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("notion-slack-bridge")

NOTION_TOKEN = "secret_..."
SLACK_TOKEN = "xoxb-..."

@mcp.tool()
async def sync_notion_to_slack(database_id: str, channel_id: str) -> str:
    """将 Notion 数据库更新同步到 Slack。
    
    Args:
        database_id: Notion 数据库 ID
        channel_id: Slack 频道 ID
    """
    # 获取 Notion 数据库记录
    async with httpx.AsyncClient() as client:
        notion_response = await client.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers={
                "Authorization": f"Bearer {NOTION_TOKEN}",
                "Notion-Version": "2022-06-28"
            },
            json={"page_size": 10}
        )
        notion_response.raise_for_status()
        pages = notion_response.json()["results"]
    
    # 发送到 Slack
    for page in pages[:5]:  # 只发送前 5 条
        title = page["properties"]["Name"]["title"][0]["plain_text"]
        status = page["properties"]["Status"]["select"]["name"]
        
        slack_message = {
            "channel": channel_id,
            "text": f"📝 新更新：*{title}* - 状态：{status}"
        }
        
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json=slack_message
        )
    
    return f"已同步 {min(5, len(pages))} 条更新到 Slack"
```

---

## 设计模式

### 1. 工具组合模式

将相关功能组织成工具组：

```python
# 文件操作工具组
@mcp.tool()
async def read_file(path: str) -> str:
    """读取文件内容。"""
    pass

@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """写入文件内容。"""
    pass

@mcp.tool()
async def list_directory(path: str) -> str:
    """列出目录内容。"""
    pass

# 使用资源提供文件 schema
@mcp.resource("file://schema")
def get_file_schema() -> str:
    """提供文件操作相关的 schema。"""
    return """
    支持的文件类型：
    - .txt, .md, .json, .yaml, .py
    最大文件大小：10MB
    """
```

### 2. 分层权限控制

```python
# 只读工具（无需认证）
@mcp.tool()
async def public_search(query: str) -> str:
    """公开搜索功能。"""
    pass

# 需要认证的工具
@mcp.tool()
async def private_data_query(user_id: str, query: str) -> str:
    """查询私有数据。"""
    # 验证用户权限
    if not verify_user(user_id):
        raise PermissionError("未授权访问")
    pass

# 管理员工具
@mcp.tool()
async def admin_operation(action: str, params: dict) -> str:
    """管理员操作。"""
    # 验证管理员权限
    if not verify_admin():
        raise PermissionError("需要管理员权限")
    pass
```

### 3. 动态工具注册

```python
class DynamicToolServer:
    def __init__(self):
        self.mcp = FastMCP("dynamic-tools")
        self.tools = {}
    
    def register_tool(self, name: str, func):
        """动态注册工具。"""
        self.tools[name] = func
        # 通知客户端工具列表已更新
        self.mcp.send_tool_list_changed()
    
    async def execute_tool(self, name: str, arguments: dict):
        """执行动态工具。"""
        if name not in self.tools:
            raise ValueError(f"未知工具：{name}")
        return await self.tools[name](**arguments)
```

---

## 最佳实践

### 1. 错误处理

```python
@mcp.tool()
async def safe_api_call(endpoint: str) -> str:
    """安全的 API 调用示例。"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.text
    
    except httpx.TimeoutException:
        return "错误：请求超时"
    except httpx.ConnectionError:
        return "错误：无法连接到服务器"
    except httpx.HTTPStatusError as e:
        return f"错误：HTTP {e.response.status_code}"
    except Exception as e:
        logging.error(f"未预期错误：{e}")
        return f"错误：{str(e)}"
```

### 2. 参数验证

```python
from pydantic import BaseModel, Field, validator

class WeatherQuery(BaseModel):
    state: str = Field(..., min_length=2, max_length=2)
    
    @validator("state")
    def validate_state_code(cls, v):
        valid_states = ["CA", "NY", "TX", "FL"]  # 示例
        if v.upper() not in valid_states:
            raise ValueError(f"无效的州代码：{v}")
        return v.upper()

@mcp.tool()
async def validated_weather(query: WeatherQuery) -> str:
    """使用 Pydantic 验证参数。"""
    # query.state 已经过验证
    pass
```

### 3. 性能优化

```python
from functools import lru_cache
import asyncio

# 缓存频繁访问的资源
@lru_cache(maxsize=100)
def get_cached_schema(schema_name: str) -> str:
    """缓存 schema 资源。"""
    return load_schema_from_db(schema_name)

# 并发处理多个请求
@mcp.tool()
async def batch_process(items: list[str]) -> str:
    """并发处理多个项目。"""
    async def process_item(item: str):
        # 异步处理逻辑
        return f"Processed: {item}"
    
    results = await asyncio.gather(*[process_item(item) for item in items])
    return "\n".join(results)
```

### 4. 日志和监控

```python
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp-server.log"),
        logging.StreamHandler(sys.stderr)
    ]
)

@mcp.tool()
async def logged_operation(param: str) -> str:
    """带日志记录的操作。"""
    logging.info(f"开始执行操作，参数：{param}")
    start_time = datetime.now()
    
    try:
        result = await perform_operation(param)
        duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"操作完成，耗时：{duration}s")
        return result
    
    except Exception as e:
        logging.error(f"操作失败：{e}", exc_info=True)
        raise
```

### 5. 配置管理

```python
import os
from pathlib import Path
from pydantic import BaseSettings

class ServerConfig(BaseSettings):
    """服务器配置。"""
    
    # API 密钥
    github_token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    notion_token: str = Field(default_factory=lambda: os.getenv("NOTION_TOKEN"))
    
    # 数据库
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL"))
    
    # 日志级别
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 使用配置
config = ServerConfig()

@mcp.tool()
async def use_configured_service() -> str:
    """使用配置的服务。"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {config.github_token}"}
        )
        return response.text
```

---

## 安全考虑

### 1. 输入验证

```python
import re

@mcp.tool()
async def safe_file_read(path: str) -> str:
    """安全的文件读取。"""
    # 防止路径遍历攻击
    if ".." in path or path.startswith("/"):
        raise ValueError("无效的文件路径")
    
    # 限制在特定目录
    base_dir = Path("/safe/directory")
    full_path = (base_dir / path).resolve()
    
    if not str(full_path).startswith(str(base_dir)):
        raise ValueError("路径超出允许范围")
    
    return full_path.read_text()
```

### 2. 敏感信息保护

```python
# ❌ 错误：硬编码密钥
API_KEY = "sk-1234567890"

# ✅ 正确：从环境变量读取
import os
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY 环境变量未设置")
```

### 3. 速率限制

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = defaultdict(list)
    
    async def check_limit(self, user_id: str) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        
        # 清理过期调用
        self.calls[user_id] = [
            t for t in self.calls[user_id] if t > cutoff
        ]
        
        if len(self.calls[user_id]) >= self.max_calls:
            return False
        
        self.calls[user_id].append(now)
        return True

rate_limiter = RateLimiter(max_calls=10, period=60)  # 每分钟 10 次

@mcp.tool()
async def rate_limited_tool(user_id: str) -> str:
    """带速率限制的工具。"""
    if not await rate_limiter.check_limit(user_id):
        raise Exception("超过速率限制，请稍后再试")
    
    # 实际工具逻辑
    pass
```

---

## 总结

### MCP 应用设计原则

1. **单一职责** - 每个服务器专注于一个领域
2. **明确接口** - 工具和资源的描述要清晰准确
3. **错误友好** - 提供有用的错误消息
4. **性能优先** - 使用异步和缓存优化性能
5. **安全第一** - 验证输入、保护密钥、限制访问

### 常见应用场景

- ✅ 企业数据集成（数据库、CRM、ERP）
- ✅ 开发工具链（Git、CI/CD、部署）
- ✅ 个人助手（日历、邮件、笔记）
- ✅ 跨平台自动化（Notion、Slack、Discord）
- ✅ AI 增强应用（搜索、分析、生成）

### 下一步

- 参考 [官方参考服务器](https://github.com/modelcontextprotocol/servers)
- 加入 [MCP 社区](https://discord.gg/modelcontextprotocol)
- 分享你的 MCP 服务器实现
