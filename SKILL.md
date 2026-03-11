# A2UI LLM Provider Refactoring Skill (Gemini to OpenAI Standard)

## 🎯 目标 (Goal)
将 A2UI (Agent-to-UI) 项目的后端 Agent 驱动模型，从深度绑定的 `google-genai` (Gemini) 彻底重构为支持自定义 Base URL 和 API Key 的 `openai` 标准格式。
**核心原则：保持所有前端渲染逻辑和底层 A2UI JSON Schema 生成逻辑绝对不变，仅替换 LLM 的通信和工具调用（Tool Calling）层。**

## 🛑 边界限制 (Scope & Constraints)
1. **绝对禁止修改的区域**：
    - `renderers/` 目录下的所有前端代码（React, Angular, Lit 等）。
    - `agent_sdks/python/src/a2ui/core/` 和 `a2ui/basic_catalog/` 目录下的核心 Schema 定义模块。
    - 所有的 `.json` 标准协议文件。
2. **必须修改的区域**：
    - `agent_sdks/python/pyproject.toml` (依赖管理)
    - `agent_sdks/python/src/a2ui/adk/` 目录 (高级 Agent 开发包封装)
    - `samples/agent/adk/` 目录下的所有后端 Agent 示例代码。

## 🛠️ 重构步骤与规则 (Refactoring Rules)

### Step 1: 替换依赖 (Dependencies)
**目标文件**：`agent_sdks/python/pyproject.toml` 和相关 `requirements.txt` / `uv.lock`。
- **动作**：移除 `google-genai` 依赖。
- **动作**：添加 `openai` (版本 >= 1.0.0) 依赖。

### Step 2: 环境变量改造 (Environment Variables)
**目标文件**：所有的 `.env.example` 和代码中读取 `GEMINI_API_KEY` 的地方。
- **动作**：将 `GEMINI_API_KEY` 替换为以下三个标准配置：
    - `OPENAI_API_KEY` (必填)
    - `OPENAI_BASE_URL` (选填，支持指向第三方网关如 OneAPI 或私有模型 vLLM/Ollama)
    - `LLM_MODEL_NAME` (必填，默认值设为 "gpt-4o" 或 "claude-3-5-sonnet-latest" 视网关而定)

### Step 3: Tool Schema 适配层 (Tool Conversion)
**背景**：A2UI 原本生成的是适用于 Gemini 的 Schema。我们需要在 `a2ui.adk` 核心中增加一个适配器，将其转换为 OpenAI 的 Function Calling 格式。
- **动作**：在调用 LLM 之前，编写一个转换函数（或在现有代码中修改），将 A2UI 提供的 `schema` 包装成 OpenAI 格式：
  ```python
  # 转换前 (Gemini 风格):
  # tool_schema = a2ui_schema 
  
  # 转换后 (OpenAI 标准):
  openai_tools = [{
      "type": "function",
      "function": {
          "name": "render_a2ui_update",  # 或者保持原名
          "description": "Render a rich user interface using A2UI components.",
          "parameters": a2ui_schema # 注入 A2UI 原生的 JSON Schema
      }
  }]
Step 4: 核心 Client 替换 (Client Initialization & Invocation)
目标文件：主要是 agent_sdks/python/src/a2ui/adk/agent.py 或执行流 agent_executor.py。

动作：将所有 from google import genai 及相关 Client 的初始化，替换为 openai.AsyncOpenAI。

模式范例：

Python
# 初始化
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
api_key=os.environ.get("OPENAI_API_KEY"),
base_url=os.environ.get("OPENAI_BASE_URL") # 关键：必须支持 base_url
)

# 调用模型
response = await client.chat.completions.create(
model=os.environ.get("LLM_MODEL_NAME", "gpt-4o"),
messages=current_messages,
tools=openai_tools,
# 如果原代码要求强制输出 UI，可使用 tool_choice
# tool_choice={"type": "function", "function": {"name": "render_a2ui_update"}}
)
Step 5: 响应解析适配 (Response Parsing)
背景：OpenAI 返回的 Tool Call 结构与 Gemini 不同。

动作：修改响应解析逻辑，从 response.choices[0].message.tool_calls 中提取 JSON arguments。

模式范例：

Python
message = response.choices[0].message
if message.tool_calls:
for tool_call in message.tool_calls:
if tool_call.function.name == "render_a2ui_update":
import json
a2ui_payload = json.loads(tool_call.function.arguments)
# ... 后续将 a2ui_payload 发送给客户端的逻辑保持不变 ...
Step 6: 样例代码全面翻新 (Samples Update)
目标文件：samples/agent/adk/*/agent.py 和 agent_executor.py。

动作：全面遍历这些示例目录（如 component_gallery, restaurant_finder, rizzcharts 等），运用 Step 4 和 Step 5 的逻辑，将示例中强绑定 Gemini 的调用代码全部重写为 OpenAI 格式。确保这些样例可以直接使用第三方网关（如运行在本地的 vLLM 或支持 OpenAI 格式的 Claude 网关）跑通。

🚀 执行指令 (Execution Command)
请你（AI Assistant）深呼吸，严格遵循以上原则，一步步执行上述 6 个步骤的重构。
每完成一个 Step，请进行内部代码校验，确保没有破坏 A2UI 的核心协议规范。如果遇到模糊地带，请默认采用最兼容 OpenAI SDK 的写法。
现在，请开始执行代码重构！