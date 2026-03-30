# MCP 实际应用场景与最佳实践

这一篇重点整理 MCP 在真实项目中的常见落地方式、设计模式和最佳实践。

这篇内容最值得读的地方，不是示例代码本身，而是它帮助你判断：哪类系统适合用 MCP 接入，工具和资源应该怎样暴露才更安全，企业、开发工具、个人助手这三类场景的设计重点分别是什么。

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

### 5. 调试技巧

**使用 MCP Inspector 调试：**

```bash
# 启动 Inspector 连接你的服务器
npx @modelcontextprotocol/inspector python my_server.py

# 指定配置文件调试
npx @modelcontextprotocol/inspector --config config.json --server my-server
```

**在代码中添加调试模式：**

```python
import sys
import json

@mcp.tool()
async def debug_tool(param: str) -> str:
    """带调试输出的工具。"""
    # 输出到 stderr，不影响 MCP 通信
    print(f"[DEBUG] 收到参数: {param}", file=sys.stderr)
    
    # 打印完整的调用栈
    import traceback
    traceback.print_stack(file=sys.stderr)
    
    return f"处理完成: {param}"

# 服务器启动时检查是否处于调试模式
if "--debug" in sys.argv:
    logging.getLogger().setLevel(logging.DEBUG)
    print("[DEBUG] 调试模式已启用", file=sys.stderr)
```

**常见调试问题及解决方案：**

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 工具不显示 | 描述不清晰 | 检查 docstring 是否完整 |
| 调用无响应 | 异步函数未 await | 确保所有 async 函数正确使用 |
| 参数解析失败 | 类型标注错误 | 使用 type hints 和 Pydantic 验证 |
| 连接断开 | 超时或阻塞 | 避免长时间阻塞操作，使用异步 I/O |

### 6. 配置管理

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

## 场景配置模板

### 1. 企业级 MCP 服务器配置

**文件：** `mcp_server_config.yaml`

```yaml
server:
  name: "enterprise-mcp-server"
  version: "1.0.0"
  description: "企业级 MCP 服务器配置"

# 服务器能力声明
capabilities:
  tools: true
  resources: true
  prompts: true
  sampling: false

# 工具配置
tools:
  database:
    enabled: true
    max_connections: 10
    timeout: 30
    retry_attempts: 3
  
  api_integration:
    enabled: true
    rate_limit: 100  # 每分钟请求数
    cache_ttl: 300   # 缓存时间（秒）

# 资源配置
resources:
  schema_files:
    path: "./schemas"
    watch_changes: true
  
  documentation:
    path: "./docs"
    format: "markdown"

# 安全配置
security:
  authentication:
    type: "bearer_token"
    token_env: "MCP_AUTH_TOKEN"
  
  authorization:
    role_based: true
    roles:
      admin: ["*"]  # 所有权限
      user: ["read", "query"]
      guest: ["read"]
  
  rate_limiting:
    enabled: true
    max_requests_per_minute: 60
    max_requests_per_hour: 1000

# 日志配置
logging:
  level: "info"
  format: "json"
  output: "./logs/mcp-server.log"
  rotation: "daily"

# 监控配置
monitoring:
  enabled: true
  metrics_port: 9090
  health_check: "/health"
```

### 2. 开发环境 MCP 配置

**文件：** `mcp_dev_config.json`

```json
{
  "server": {
    "name": "dev-mcp-server",
    "debug": true,
    "hot_reload": true
  },
  "tools": {
    "code_analysis": {
      "enabled": true,
      "languages": ["python", "javascript", "typescript"],
      "linters": {
        "python": "ruff",
        "javascript": "eslint"
      }
    },
    "git_integration": {
      "enabled": true,
      "auto_commit": false,
      "commit_message_template": "feat: {description}"
    },
    "test_runner": {
      "enabled": true,
      "framework": "pytest",
      "coverage_threshold": 80
    }
  },
  "resources": {
    "project_files": {
      "include": ["**/*.py", "**/*.js", "**/*.ts", "**/*.md"],
      "exclude": ["node_modules/**", "__pycache__/**", ".git/**"]
    }
  },
  "development": {
    "auto_reload": true,
    "verbose_logging": true,
    "mock_external_apis": true
  }
}
```

### 3. 个人助手 MCP 配置

**文件：** `personal_assistant_config.yaml`

```yaml
server:
  name: "personal-assistant"
  description: "个人 AI 助手 MCP 服务器"

# 个人数据源配置
data_sources:
  calendar:
    provider: "google_calendar"
    scopes: ["https://www.googleapis.com/auth/calendar.readonly"]
    sync_interval: 300  # 5分钟同步一次
  
  email:
    provider: "gmail"
    scopes: ["https://www.googleapis.com/auth/gmail.readonly"]
    max_emails: 50
  
  notes:
    provider: "notion"
    databases:
      - name: "任务列表"
        id: "${NOTION_TASKS_DB_ID}"
      - name: "笔记"
        id: "${NOTION_NOTES_DB_ID}"
  
  files:
    provider: "local"
    paths:
      - "~/Documents"
      - "~/Projects"
    file_types: [".md", ".txt", ".pdf", ".docx"]

# 工具配置
tools:
  scheduling:
    enabled: true
    default_reminder: 15  # 提前15分钟提醒
  
  search:
    enabled: true
    engines: ["google", "bing"]
    max_results: 10
  
  translation:
    enabled: true
    default_source: "auto"
    default_target: "zh-CN"

# 隐私和安全
privacy:
  data_retention: "30d"
  encryption: true
  local_processing: true
  no_cloud_storage: true

# 个性化设置
personalization:
  language: "zh-CN"
  timezone: "Asia/Shanghai"
  working_hours:
    start: "09:00"
    end: "18:00"
    days: ["monday", "tuesday", "wednesday", "thursday", "friday"]
```

### 4. 配置最佳实践

#### 环境变量管理

```bash
# .env 文件示例
MCP_SERVER_NAME=my-mcp-server
MCP_AUTH_TOKEN=your-secret-token
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379

# API 密钥
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
NOTION_API_KEY=secret_xxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

#### 配置加载策略

```python
import os
from pathlib import Path
import yaml
import json

class MCPConfig:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _find_config(self) -> Path:
        """按优先级查找配置文件"""
        search_paths = [
            Path("./mcp_config.yaml"),
            Path("./mcp_config.json"),
            Path.home() / ".config" / "mcp" / "config.yaml",
            Path("/etc/mcp/config.yaml")
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError("未找到 MCP 配置文件")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path) as f:
            if self.config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        # 支持 MCP_ 前缀的环境变量
        for key, value in os.environ.items():
            if key.startswith("MCP_"):
                config_key = key[4:].lower().replace("_", ".")
                self._set_nested(self.config, config_key, value)
    
    def _set_nested(self, obj: dict, key: str, value: str):
        """设置嵌套配置值"""
        keys = key.split(".")
        for k in keys[:-1]:
            obj = obj.setdefault(k, {})
        obj[keys[-1]] = value
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
```

#### 配置验证

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class SecurityConfig(BaseModel):
    authentication_type: str = "bearer_token"
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = Field(ge=1, le=1000, default=60)

class ServerConfig(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = None
    debug: bool = False

class MCPConfigSchema(BaseModel):
    server: ServerConfig
    security: SecurityConfig = SecurityConfig()
    log_level: LogLevel = LogLevel.INFO
    
    class Config:
        validate_assignment = True
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
