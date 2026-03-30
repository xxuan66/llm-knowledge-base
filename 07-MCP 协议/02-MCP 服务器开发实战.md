# MCP 服务器开发实战

本教程将带你从零开始构建一个完整的 MCP 服务器，并连接到 AI 宿主应用。

这篇内容最适合的阅读方式不是“复制代码”，而是理解一个 MCP Server 最小闭环到底由哪些部分组成：

- 工具定义
- 输入输出约束
- 外部 API 调用
- 错误处理
- 调试与日志

## 项目概述

我们将构建一个**天气查询 MCP 服务器**，提供两个工具：
- `get_alerts` - 获取某州的天气警报
- `get_forecast` - 获取指定地点的天气预报

完成后，你的 MCP 服务器可以连接到 Claude Desktop、VS Code 等任何支持 MCP 的客户端。

---

## 做 MCP Server 时要特别注意什么

1. 只暴露必要能力，不要把所有内部接口都直接开放给模型
2. 工具描述必须清晰，否则模型很难正确调用
3. 敏感操作要有额外确认机制
4. 日志要可追踪，方便调试模型为什么会调用失败

## 前置准备

### 需要的知识

- Python 基础
- 对 LLM（如 Claude）有基本了解

### 系统要求

- Python 3.10 或更高版本
- Python MCP SDK 1.2.0 或更高版本

---

## 第一步：环境搭建

### 1. 安装 uv（Python 包管理工具）

**macOS/Linux：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows：**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> ⚠️ 安装后请重启终端以确保 `uv` 命令可用

### 2. 创建项目

```bash
# 创建新项目目录
uv init weather
cd weather

# 创建虚拟环境并激活
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
uv add "mcp[cli]" httpx

# 创建服务器文件
touch weather.py  # Windows: new-item weather.py
```

### 3. 项目结构

```
weather/
├── .venv/              # 虚拟环境
├── weather.py          # 主服务器代码
├── pyproject.toml      # 项目配置
└── .git/
```

---

## 第二步：编写 MCP 服务器

### 完整代码

创建 `weather.py` 文件：

```python
from typing import Any
import sys
import logging

import httpx
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服务器
mcp = FastMCP("weather")

# 常量
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


# ========== 辅助函数 ==========

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """向 NWS API 发送请求，包含错误处理。"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Request failed: {e}")
            return None


def format_alert(feature: dict) -> str:
    """将警报特征格式化为可读字符串。"""
    props = feature["properties"]
    return f"""
事件：{props.get("event", "未知")}
区域：{props.get("areaDesc", "未知")}
严重程度：{props.get("severity", "未知")}
描述：{props.get("description", "暂无描述")}
指示：{props.get("instruction", "暂无具体指示")}
"""


# ========== MCP 工具定义 ==========

@mcp.tool()
async def get_alerts(state: str) -> str:
    """获取美国某州的天气警报。

    Args:
        state: 两位字母的美国州代码（如 CA、NY）
    
    Returns:
        格式化的警报信息字符串
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "无法获取警报数据或未找到警报。"

    if not data["features"]:
        return "该州当前无活跃警报。"

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """获取指定地点的天气预报。

    Args:
        latitude: 地点纬度
        longitude: 地点经度
    
    Returns:
        格式化的天气预报信息
    """
    # 首先获取预报网格端点
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "无法获取该地点的预报数据。"

    # 从 points 响应中获取预报 URL
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "无法获取详细预报。"

    # 格式化周期为可读预报
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # 只显示未来 5 个时段
        forecast = f"""
{period["name"]}:
温度：{period["temperature"]}°{period["temperatureUnit"]}
风力：{period["windSpeed"]} {period["windDirection"]}
预报：{period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


# ========== 服务器入口 ==========

def main():
    """初始化并运行服务器。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]  # 必须输出到 stderr
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

---

## 第三步：理解代码结构

### 1. 导入和初始化

```python
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("weather")
```

- `FastMCP` 是 MCP SDK 提供的高级抽象
- 自动根据函数签名和文档字符串生成工具定义
- 服务器名称为 "weather"

### 2. 工具装饰器

```python
@mcp.tool()
async def get_alerts(state: str) -> str:
    """工具描述文档..."""
```

- `@mcp.tool()` 装饰器将函数注册为 MCP 工具
- 函数文档字符串（docstring）作为工具描述
- 参数类型提示用于自动生成参数 schema

### 3. 异步处理

```python
async def make_nws_request(url: str) -> dict[str, Any] | None:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30.0)
```

- 使用 `async/await` 处理 I/O 操作
- `httpx.AsyncClient` 提供高效的异步 HTTP 请求

### 4. 错误处理

```python
try:
    response.raise_for_status()
    return response.json()
except Exception as e:
    logging.error(f"Request failed: {e}")
    return None
```

- 捕获所有异常并返回 `None`
- 使用 `logging.error` 记录错误（输出到 stderr）

---

## 第四步：运行和测试服务器

### 1. 直接运行服务器

```bash
uv run weather.py
```

服务器启动后会等待来自 MCP 宿主的连接消息。

### 2. 使用 MCP Inspector 测试

MCP 官方提供了调试工具：

```bash
# 安装 MCP Inspector
npm install -g @modelcontextprotocol/inspector

# 启动 Inspector
npx @modelcontextprotocol/inspector uv run weather.py
```

Inspector 会打开一个 Web 界面（通常在 http://localhost:5173），可以：
- 查看可用工具列表
- 手动调用工具测试
- 查看 JSON-RPC 消息日志

---

## 第五步：连接到 Claude Desktop

### 1. 找到配置文件

**macOS/Linux：**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows：**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### 2. 配置 MCP 服务器

编辑配置文件，添加：

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/weather",
        "run",
        "weather.py"
      ]
    }
  }
}
```

> ⚠️ **重要：**
> - 使用**绝对路径**到你的项目目录
> - Windows 路径使用双反斜杠 `\\` 或正斜杠 `/`
> - 可能需要指定 `uv` 的完整路径（用 `which uv` 查找）

### 3. 重启 Claude Desktop

保存配置文件后重启 Claude Desktop，MCP 功能会自动激活。

### 4. 测试使用

在 Claude Desktop 中尝试：

```
帮我查询加州的天气警报
```

或

```
纽约市（纬度 40.7128，经度 -74.0060）的天气预报是什么？
```

---

## 关键注意事项

### ⚠️ 日志处理（非常重要！）

**对于 STDIO 传输的服务器：**

❌ **错误做法：**
```python
print("Processing request")  # 会破坏 JSON-RPC 消息！
console.log("Server started")  # TypeScript 版本
```

✅ **正确做法：**
```python
# Python - 输出到 stderr
print("Processing request", file=sys.stderr)
logging.info("Processing request")  # 配置为输出到 stderr

# TypeScript - 输出到 stderr
console.error("Server started");
```

**原因：** stdout 用于 JSON-RPC 通信，任何额外输出都会破坏协议

### ✅ 最佳实践

1. **使用日志库** - 如 Python 的 `logging` 模块
2. **始终输出到 stderr** - 避免污染 stdout
3. **结构化错误处理** - 捕获异常并返回友好错误消息
4. **添加详细文档** - 工具描述和参数说明要清晰

---

## 扩展项目

### 添加 Resource（资源）

```python
@mcp.resource("weather://schema")
def get_schema() -> str:
    """提供 API schema 作为资源。"""
    return """
    {
      "type": "object",
      "properties": {
        "state": {"type": "string"},
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      }
    }
    """
```

### 添加 Prompt（提示）

```python
@mcp.prompt()
def weather_query_template(location: str) -> str:
    """生成天气查询提示模板。"""
    return f"""
    你是一个天气助手。请根据以下位置提供天气信息：
    位置：{location}
    
    请查询：
    1. 当前天气状况
    2. 未来 5 天预报
    3. 任何活跃的天气警报
    """
```

---

## 调试技巧

### 1. 启用详细日志

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 2. 使用 MCP Inspector

- 查看完整的 JSON-RPC 消息流
- 手动测试工具调用
- 检查错误详情

### 3. 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Claude 不显示 MCP 工具 | 配置路径错误 | 检查绝对路径，重启 Claude |
| 工具调用失败 | API 密钥/网络问题 | 检查错误日志，测试 API 访问 |
| 服务器无响应 | stdout 被污染 | 确保所有输出到 stderr |

---

## 总结

你现在已经完成了：

✅ 搭建 Python MCP 开发环境  
✅ 编写包含两个工具的 MCP 服务器  
✅ 理解 MCP 工具、资源、提示的定义方式  
✅ 配置 Claude Desktop 连接服务器  
✅ 掌握日志处理和调试技巧  

**下一步：** 学习如何 [构建 MCP 客户端应用](./03-MCP 客户端开发指南.md)

---

## 参考资源

- [MCP 官方文档](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [参考服务器实现](https://github.com/modelcontextprotocol/servers)
