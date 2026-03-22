# MCP 客户端开发指南

本教程将带你构建一个 MCP 客户端应用，连接到 MCP 服务器并与之交互。

## 项目概述

我们将使用 Python 构建一个 MCP 客户端，能够：
- 连接到本地或远程 MCP 服务器
- 发现服务器提供的工具、资源和提示
- 调用服务器工具并处理响应
- 实现自定义的 MCP Host 应用

---

## 前置准备

### 系统要求

- Python 3.10 或更高版本
- Python MCP SDK 1.2.0 或更高版本

### 环境搭建

```bash
# 创建项目目录
mkdir mcp-client
cd mcp-client

# 初始化项目
uv init
uv venv
source .venv/bin/activate

# 安装依赖
uv add mcp
```

---

## 第一步：理解客户端架构

### MCP Host 的职责

作为 MCP Host（如 Claude Desktop、VS Code），你的应用需要：

1. **管理连接** - 建立和维护与 MCP 服务器的连接
2. **能力协商** - 在初始化时交换支持的功能
3. **发现原语** - 获取服务器提供的工具、资源、提示列表
4. **执行操作** - 调用工具、读取资源、获取提示
5. **处理通知** - 接收服务器的实时更新

### 客户端工作流程

```
1. 创建客户端实例
       ↓
2. 连接到服务器 (Stdio/HTTP)
       ↓
3. 发送 initialize 请求
       ↓
4. 获取能力列表 (tools/resources/prompts)
       ↓
5. 调用工具/读取资源
       ↓
6. 处理响应和通知
```

---

## 第二步：构建基础客户端

### 完整代码示例

创建 `client.py`：

```python
import asyncio
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def create_client_session(script_path: str):
    """创建与 MCP 服务器的会话连接。"""
    
    # 配置服务器参数
    server_params = StdioServerParameters(
        command="uv",
        args=["run", script_path],
        cwd=None,
    )
    
    # 建立 stdio 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()
            yield session


async def main():
    """主函数：演示客户端功能。"""
    
    # 服务器脚本路径（替换为你的 MCP 服务器路径）
    server_script = "/path/to/weather.py"
    
    async with create_client_session(server_script) as session:
        print("✅ 已成功连接到 MCP 服务器")
        
        # ========== 1. 列出可用工具 ==========
        tools_result = await session.list_tools()
        print(f"\n📦 可用工具 ({len(tools_result.tools)} 个):")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # ========== 2. 调用工具 ==========
        print("\n🔧 调用工具示例:")
        
        # 调用 get_alerts 工具
        alerts_result = await session.call_tool(
            "get_alerts",
            arguments={"state": "CA"}
        )
        print(f"\n加州天气警报:\n{alerts_result.content[0].text}")
        
        # 调用 get_forecast 工具
        forecast_result = await session.call_tool(
            "get_forecast",
            arguments={"latitude": 40.7128, "longitude": -74.0060}
        )
        print(f"\n纽约天气预报:\n{forecast_result.content[0].text}")
        
        # ========== 3. 列出资源 ==========
        try:
            resources_result = await session.list_resources()
            print(f"\n📚 可用资源 ({len(resources_result.resources)} 个):")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")
        except Exception as e:
            print(f"\n资源列表不可用：{e}")
        
        # ========== 4. 列出提示 ==========
        try:
            prompts_result = await session.list_prompts()
            print(f"\n💡 可用提示 ({len(prompts_result.prompts)} 个):")
            for prompt in prompts_result.prompts:
                print(f"  - {prompt.name}: {prompt.description}")
        except Exception as e:
            print(f"\n提示列表不可用：{e}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 第三步：代码详解

### 1. 建立连接

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="uv",
    args=["run", "weather.py"],
    cwd="/path/to/project",
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

**关键点：**
- `StdioServerParameters` 定义如何启动服务器进程
- `stdio_client` 建立标准输入/输出连接
- `ClientSession` 管理会话生命周期
- `initialize()` 执行能力协商

### 2. 发现工具

```python
tools_result = await session.list_tools()

for tool in tools_result.tools:
    print(f"工具名：{tool.name}")
    print(f"描述：{tool.description}")
    print(f"参数 schema: {tool.inputSchema}")
```

**Tool 对象属性：**
- `name` - 工具名称
- `description` - 工具描述
- `inputSchema` - JSON Schema 格式的参数定义

### 3. 调用工具

```python
result = await session.call_tool(
    "get_alerts",
    arguments={"state": "CA"}
)

# 处理响应
for content in result.content:
    if hasattr(content, 'text'):
        print(content.text)
```

**响应结构：**
- `content` - 内容列表（可能包含文本、图像等）
- `isError` - 是否发生错误

### 4. 错误处理

```python
try:
    result = await session.call_tool("tool_name", arguments={})
except Exception as e:
    print(f"工具调用失败：{e}")
```

---

## 第四步：构建交互式客户端

### 支持用户输入的完整应用

创建 `interactive_client.py`：

```python
import asyncio
import json
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPInteractiveClient:
    """交互式 MCP 客户端。"""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session: ClientSession | None = None
    
    @asynccontextmanager
    async def connect(self):
        """连接到 MCP 服务器。"""
        server_params = StdioServerParameters(
            command="uv",
            args=["run", self.server_script],
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                yield self
                self.session = None
    
    async def show_menu(self):
        """显示主菜单。"""
        print("\n" + "="*50)
        print("MCP 客户端菜单")
        print("="*50)
        print("1. 列出工具")
        print("2. 调用工具")
        print("3. 列出资源")
        print("4. 读取资源")
        print("5. 列出提示")
        print("6. 获取提示")
        print("0. 退出")
        print("="*50)
    
    async def list_tools(self):
        """列出所有可用工具。"""
        if not self.session:
            print("❌ 未连接到服务器")
            return
        
        result = await self.session.list_tools()
        print(f"\n找到 {len(result.tools)} 个工具:")
        for i, tool in enumerate(result.tools, 1):
            print(f"\n{i}. {tool.name}")
            print(f"   描述：{tool.description}")
            print(f"   参数：{json.dumps(tool.inputSchema, indent=2)}")
    
    async def call_tool_interactive(self):
        """交互式调用工具。"""
        if not self.session:
            print("❌ 未连接到服务器")
            return
        
        # 获取工具列表
        tools_result = await self.session.list_tools()
        tools = tools_result.tools
        
        if not tools:
            print("❌ 没有可用工具")
            return
        
        # 显示工具列表
        print("\n可用工具:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name} - {tool.description}")
        
        # 选择工具
        try:
            choice = int(input("\n选择工具编号：")) - 1
            if choice < 0 or choice >= len(tools):
                print("❌ 无效选择")
                return
            
            tool = tools[choice]
            print(f"\n调用工具：{tool.name}")
            
            # 解析参数 schema
            schema = tool.inputSchema
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # 收集参数
            arguments = {}
            for param_name, param_schema in properties.items():
                is_required = param_name in required
                param_type = param_schema.get("type", "string")
                
                prompt = f"输入 {param_name}"
                if is_required:
                    prompt += " (必填)"
                prompt += f" [{param_type}]: "
                
                value = input(prompt)
                if not value and not is_required:
                    continue
                
                # 类型转换
                if param_type == "number":
                    value = float(value)
                elif param_type == "integer":
                    value = int(value)
                elif param_type == "boolean":
                    value = value.lower() in ("true", "yes", "1")
                
                arguments[param_name] = value
            
            # 调用工具
            print(f"\n执行工具...")
            result = await self.session.call_tool(tool.name, arguments)
            
            # 显示结果
            print("\n结果:")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)
                else:
                    print(content)
            
            if result.isError:
                print("⚠️ 工具执行返回错误")
        
        except ValueError as e:
            print(f"❌ 输入错误：{e}")
        except Exception as e:
            print(f"❌ 调用失败：{e}")
    
    async def list_resources(self):
        """列出所有资源。"""
        if not self.session:
            print("❌ 未连接到服务器")
            return
        
        result = await self.session.list_resources()
        print(f"\n找到 {len(result.resources)} 个资源:")
        for resource in result.resources:
            print(f"  URI: {resource.uri}")
            print(f"  名称：{resource.name}")
            print(f"  描述：{resource.description}")
            print()
    
    async def list_prompts(self):
        """列出所有提示。"""
        if not self.session:
            print("❌ 未连接到服务器")
            return
        
        result = await self.session.list_prompts()
        print(f"\n找到 {len(result.prompts)} 个提示:")
        for prompt in result.prompts:
            print(f"  名称：{prompt.name}")
            print(f"  描述：{prompt.description}")
            print()
    
    async def run(self):
        """运行交互式客户端。"""
        async with self.connect():
            print("✅ 已连接到 MCP 服务器")
            
            while True:
                await self.show_menu()
                choice = input("请选择操作：")
                
                if choice == "1":
                    await self.list_tools()
                elif choice == "2":
                    await self.call_tool_interactive()
                elif choice == "3":
                    await self.list_resources()
                elif choice == "4":
                    print("资源读取功能待实现")
                elif choice == "5":
                    await self.list_prompts()
                elif choice == "6":
                    print("提示获取功能待实现")
                elif choice == "0":
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择")


async def main():
    # 替换为你的 MCP 服务器路径
    server_script = "/path/to/weather.py"
    
    client = MCPInteractiveClient(server_script)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 第五步：连接远程 HTTP 服务器

### 使用 Streamable HTTP 传输

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def connect_to_remote_server(url: str, auth_token: str = None):
    """连接到远程 HTTP MCP 服务器。"""
    
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    async with streamablehttp_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 现在可以像平常一样使用 session
            tools = await session.list_tools()
            print(f"找到 {len(tools.tools)} 个工具")
            
            return session
```

---

## 第六步：实现通知处理

### 订阅工具变更通知

```python
from mcp.types import ToolListChangedNotification


async def setup_notification_handlers(session: ClientSession):
    """设置通知处理器。"""
    
    # 注册工具列表变更通知处理器
    @session.notification_handler
    async def handle_tool_list_changed(notification: ToolListChangedNotification):
        print("🔔 工具列表已更新！")
        # 重新获取工具列表
        tools = await session.list_tools()
        print(f"当前可用工具：{len(tools.tools)} 个")
    
    print("✅ 通知处理器已设置")
```

---

## 第七步：构建自定义 Host 应用

### 集成到 Web 应用

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict

class MCPService:
    def __init__(self):
        self.session = None
    
    async def connect(self, server_script: str):
        # 连接逻辑...
        pass
    
    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.session:
            raise Exception("未连接到服务器")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result

mcp_service = MCPService()

@app.on_event("startup")
async def startup_event():
    await mcp_service.connect("/path/to/weather.py")

@app.post("/tools/{tool_name}")
async def invoke_tool(tool_name: str, request: ToolCallRequest):
    try:
        result = await mcp_service.call_tool(tool_name, request.arguments)
        return {"success": True, "result": result.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 调试技巧

### 1. 启用详细日志

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. 捕获原始消息

```python
from mcp.client.stdio import stdio_client

async with stdio_client(server_params) as (read, write):
    # 读取原始消息
    async for message in read:
        print(f"收到消息：{message}")
```

### 3. 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Connection refused | 服务器未启动 | 检查服务器路径和命令 |
| Initialize failed | 协议版本不匹配 | 更新 MCP SDK |
| Tool not found | 工具名错误 | 先调用 list_tools 确认 |

---

## 总结

你现在已经掌握了：

✅ 使用 Python MCP SDK 构建客户端  
✅ 连接本地和远程 MCP 服务器  
✅ 发现和调用服务器工具  
✅ 处理资源和提示  
✅ 实现通知处理  
✅ 构建交互式客户端应用  

**关键要点：**
- 使用 `ClientSession` 管理连接
- 始终先调用 `initialize()` 进行能力协商
- 使用 `list_*` 方法发现可用功能
- 正确处理异步操作和错误

---

## 参考资源

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP 规范](https://modelcontextprotocol.io/specification)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
