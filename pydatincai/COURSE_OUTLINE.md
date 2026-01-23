# PydanticAI Agent 开发：从入门到精通

> 本课程采用 "In Action" 方式组织，每个主题都通过实际项目来学习，强调动手实践而非理论讲解。
> 
> ⚠️ **API 版本说明**：本课程基于 PydanticAI 最新版本，注意以下重要变化：
> - `result_type` → `output_type`
> - `system_prompt` → `instructions`（推荐）
> - `result.data` → `result.output`
>
> 📚 **官方文档**：https://ai.pydantic.dev/
> 📄 **LLMs.txt**：https://ai.pydantic.dev/llms.txt

## 课程理念

- 🎯 **项目驱动**：每个章节都有一个实际项目
- 🔨 **动手为主**：80% 实践 + 20% 理论
- 🪜 **循序渐进**：从简单到复杂，螺旋式上升
- 🎓 **真实场景**：所有项目都来自实际业务需求

---

## Part 1: 基础篇 - 构建你的第一个 Agent

### Chapter 1: Hello Agent World
**项目：构建一个简单的问答助手**

> 官方文档：[Agents](https://ai.pydantic.dev/agents/index.md) | [Installation](https://ai.pydantic.dev/install/index.md)

- [ ] 1.1 环境搭建
  - 安装 PydanticAI：`pip install pydantic-ai`
  - 配置 LLM（支持多种模型：OpenAI、Anthropic、Google、Groq 等）
  - 理解模型字符串格式：`provider:model-name`
  - 第一个 Agent 程序

- [ ] 1.2 理解 Agent 的核心组件
  - Agent 是什么？
  - `instructions` 参数：静态指令 vs 动态指令
  - 三种运行方式：`run()`、`run_sync()`、`run_stream()`
  - 理解 `result.output` 的返回值

- [ ] 1.3 支持的模型提供商
  - OpenAI / Azure OpenAI
  - Anthropic (Claude)
  - Google (Gemini)
  - Groq / Mistral / Cohere
  - Bedrock / HuggingFace
  - OpenRouter / Cerebras
  - 本地模型：Outlines

- [ ] 1.4 实践：构建天气查询助手
  - 从简单对话开始
  - 添加第一个工具函数
  - 观察 Agent 的决策过程（使用 `result.all_messages()`）

**输出：** 一个能够回答天气查询的简单 Agent

---

### Chapter 2: 工具函数 - Agent 的手和脚
**项目：ClickHouse 表结构助手**

> 官方文档：[Function Tools](https://ai.pydantic.dev/tools/index.md) | [Dependencies](https://ai.pydantic.dev/dependencies/index.md)

- [x] 2.1 什么是工具函数
  - 工具函数的本质：让 LLM 能够调用外部功能
  - `@agent.tool` vs `@agent.tool_plain` 装饰器
  - 函数签名、类型注解和文档字符串的重要性
  - 参数描述自动从 docstring 提取

- [x] 2.2 设计工具函数
  - 单一职责原则
  - 参数设计和类型注解（Pydantic 验证）
  - 返回值的选择（JSON 可序列化类型）

- [x] 2.3 依赖注入
  - 为什么需要依赖注入？
  - `deps_type` 和 `RunContext`
  - 实践：管理数据库连接

- [ ] 2.4 工具函数的高级模式
  - `Tool` 类：更精细的工具定义控制
  - 错误处理：`ModelRetry` 异常
  - 工具超时设置

- [ ] 2.5 工具设计的权衡
  - 细粒度 vs 通用工具
  - 安全性考虑（SQL 注入等）
  - 实践：添加 `find_tables_with_column` 工具

**输出：** 一个功能完整的数据库结构查询 Agent

---

### Chapter 3: 高级工具特性
**项目：多模态工具助手**

> 官方文档：[Advanced Tool Features](https://ai.pydantic.dev/tools-advanced/index.md)

- [ ] 3.1 工具返回多模态内容
  - `ImageUrl` - 返回图片
  - `DocumentUrl` - 返回文档
  - `AudioUrl` / `VideoUrl` - 返回音视频

- [ ] 3.2 ToolReturn - 精细控制返回值
  - `return_value`：程序使用的返回值
  - `content`：提供给模型的上下文
  - `metadata`：应用元数据（类似其他框架的 "artifacts"）

- [ ] 3.3 Tool.from_schema()
  - 为文档不佳的函数创建工具
  - 自定义 JSON Schema

- [ ] 3.4 实践：构建文件分析助手
  - 返回图片预览
  - 返回文档摘要
  - 元数据追踪

**输出：** 一个能处理多种文件类型的分析 Agent

---

### Chapter 4: 内置工具与通用工具
**项目：多功能信息检索助手**

> 官方文档：[Built-in Tools](https://ai.pydantic.dev/builtin-tools/index.md) | [Common Tools](https://ai.pydantic.dev/common-tools/index.md)

- [ ] 4.1 WebFetchTool - 网页抓取
  - 配置 `allowed_domains` 和 `blocked_domains`
  - `max_uses` 和 `max_content_tokens` 限制
  - 启用引用 `enable_citations`

- [ ] 4.2 ImageGenerationTool - 图像生成
  - 与 OpenAI DALL-E 集成
  - 处理 `BinaryImage` 返回值

- [ ] 4.3 MemoryTool - 对话记忆
  - 实现持久化记忆存储
  - 自定义记忆后端（数据库、云存储）

- [ ] 4.4 CodeExecutionTool - 代码执行
  - 安全沙箱执行
  - 输入输出处理

- [ ] 4.5 Common Tools - 通用工具
  - 预定义的常用工具集
  - 快速集成常见功能

- [ ] 4.6 实践：构建多功能研究助手
  - 组合多个内置工具
  - 工具选择策略

**输出：** 一个能够搜索网页、生成图像的多功能助手

---

### Chapter 5: 多模态输入
**项目：多模态内容分析助手**

> 官方文档：[Image, Audio, Video & Document Input](https://ai.pydantic.dev/input/index.md)

- [ ] 5.1 图片输入
  - 支持的格式和大小限制
  - Base64 vs URL 输入
  - 图片描述和分析

- [ ] 5.2 音频输入
  - 音频转录
  - 语音理解

- [ ] 5.3 视频输入
  - 视频帧提取
  - 视频内容分析

- [ ] 5.4 文档输入
  - PDF 处理
  - 文档结构理解

- [ ] 5.5 实践：构建多模态分析助手
  - 图片内容识别
  - 文档信息提取
  - 音视频转录和分析

**输出：** 一个能处理多种媒体类型的分析 Agent

---

### Chapter 6: 对话管理与消息历史
**项目：多轮对话的客服助手**

> 官方文档：[Messages and chat history](https://ai.pydantic.dev/message-history/index.md)

- [x] 6.1 理解 LLM 的无状态本质
  - 为什么需要 message_history？
  - 对话历史的结构：`ModelRequest`、`ModelResponse`

- [ ] 6.2 实现对话记忆
  - 使用 `message_history` 参数
  - 从上一次运行获取：`result.all_messages()`
  - 只获取新消息：`result.new_messages()`

- [ ] 6.3 上下文引用
  - "上一个"、"那个表" 的处理
  - 代词解析和指代消解

- [ ] 6.4 多轮对话的设计模式
  - 澄清型问题
  - 引导式对话
  - 任务分解

**输出：** 一个能够进行自然多轮对话的客服 Agent

---

## Part 2: 进阶篇 - 构建生产级 Agent

### Chapter 7: 结构化输出
**项目：数据提取和结构化 Agent**

> 官方文档：[Output](https://ai.pydantic.dev/output/index.md)

- [ ] 7.1 为什么需要结构化输出？
  - 自由文本 vs 结构化数据
  - 使用场景

- [ ] 7.2 使用 Pydantic 定义输出结构
  - `output_type` 参数（注意：旧版为 `result_type`）
  - 复杂嵌套结构
  - 列表和可选字段
  - Union 类型处理

- [ ] 7.3 输出函数（Output Functions）
  - `@agent.output` 装饰器
  - 作为最终动作执行，结果不返回给模型
  - 与普通工具的区别

- [ ] 7.4 实践：从自然语言提取结构化数据
  - 解析用户意图
  - 提取实体和关系
  - 验证和错误处理（`output_retries`）

- [ ] 7.5 结构化输出的高级用法
  - 条件字段
  - 流式结构化输出
  - 部分结果处理

**输出：** 一个能从非结构化文本中提取结构化信息的 Agent

---

### Chapter 8: 流式响应
**项目：实时翻译和总结 Agent**

- [ ] 8.1 为什么需要流式响应？
  - 用户体验的重要性
  - 长文本处理

- [ ] 8.2 实现流式输出
  - `agent.run_stream()` 方法
  - 事件类型：`PartStartEvent`、`PartDeltaEvent`、`FinalResultEvent`
  - `TextPartDelta`、`ToolCallPartDelta`、`ThinkingPartDelta`
  - 在终端中显示流式响应

- [ ] 8.3 流式响应中的工具调用
  - `FunctionToolCallEvent` 和 `FunctionToolResultEvent`
  - 工具调用时的用户提示
  - 多次工具调用的处理

- [ ] 8.4 实践：流式文档翻译
  - 分块处理长文档
  - 实时显示翻译进度
  - 保持上下文连贯性

**输出：** 一个带有实时反馈的翻译 Agent

---

### Chapter 9: Thinking - 模型思考过程
**项目：可解释的推理助手**

> 官方文档：[Thinking](https://ai.pydantic.dev/thinking/index.md)

- [ ] 9.1 什么是 Thinking？
  - Chain of Thought (CoT) 的价值
  - 模型内部推理过程

- [ ] 9.2 启用和访问思考过程
  - 配置模型输出思考内容
  - `ThinkingPartDelta` 事件
  - 思考内容的结构化

- [ ] 9.3 思考过程的应用
  - 调试 Agent 决策
  - 提升可解释性
  - 用户信任建设

- [ ] 9.4 实践：构建可解释的决策助手
  - 展示推理步骤
  - 解释工具选择原因
  - 透明的错误分析

**输出：** 一个展示完整推理过程的可解释 Agent

---

### Chapter 10: 直接模型请求
**项目：灵活的模型调用系统**

> 官方文档：[Direct Model Requests](https://ai.pydantic.dev/direct/index.md)

- [ ] 10.1 为什么需要直接模型请求？
  - 绕过 Agent 框架的场景
  - 更精细的控制需求

- [ ] 10.2 直接调用模型
  - `model.request()` 方法
  - 手动构建消息
  - 处理原始响应

- [ ] 10.3 与 Agent 的对比
  - 何时使用 Agent
  - 何时直接调用模型
  - 混合使用策略

- [ ] 10.4 实践：构建灵活的模型调用层
  - 统一的模型接口
  - 批量请求处理
  - 响应后处理

**输出：** 一个灵活的模型调用封装层

---

### Chapter 11: HTTP 重试与错误处理
**项目：健壮的 Agent 系统**

> 官方文档：[HTTP Request Retries](https://ai.pydantic.dev/retries/index.md)

- [ ] 11.1 常见的 Agent 错误
  - LLM 幻觉
  - 工具调用失败
  - 网络超时

- [ ] 11.2 HTTP 重试机制
  - 自动重试配置
  - 指数退避策略
  - 重试条件定制

- [ ] 11.3 优雅的错误处理
  - `ModelRetry` 异常：告诉模型重试
  - 重试配置：`retries` 参数
  - 访问重试次数：`ctx.retry`
  - 降级策略

- [ ] 11.4 用户输入验证
  - Pydantic 自动验证工具参数
  - 验证错误自动传递给模型重试
  - 防止注入攻击
  - 安全最佳实践

**输出：** 一个具有完善错误处理的生产级 Agent

---

### Chapter 12: Human-in-the-Loop - 工具审批机制
**项目：带审批流程的自动化助手**

- [ ] 12.1 为什么需要人工审批？
  - 高风险操作的安全控制
  - 合规要求
  - 用户信任建设

- [ ] 12.2 实现工具审批
  - 工具的 `require_approval` 参数
  - 根据参数条件决定是否需要审批
  - `DeferredToolRequests` 处理

- [ ] 12.3 审批工作流
  - 前端展示待审批操作
  - 用户确认/拒绝
  - 使用 `DeferredToolResults` 继续执行

- [ ] 12.4 实践：构建安全的文件操作助手
  - 删除、修改操作需要审批
  - 读取操作自动执行
  - 审批历史记录

**输出：** 一个带有安全审批机制的文件管理 Agent

---

### Chapter 13: Agent 测试
**项目：为 ClickHouse 助手编写测试**

> 官方文档：[Testing](https://ai.pydantic.dev/testing/index.md)

- [ ] 13.1 为什么测试 Agent 很难？
  - LLM 的不确定性
  - 测试策略

- [ ] 13.2 单元测试工具函数
  - Mock 依赖
  - 测试边界条件
  - 使用 pytest

- [ ] 13.3 集成测试 Agent
  - 使用 `TestModel` 固定 LLM 输出
  - `FunctionModel` 自定义响应
  - 验证工具调用序列
  - 端到端测试

- [ ] 13.4 调试技巧
  - 追踪 Agent 的决策过程
  - 分析失败案例
  - 日志记录最佳实践

**输出：** 一套完整的测试用例

---

### Chapter 14: 调试与监控
**项目：可观测的 Agent 系统**

> 官方文档：[Debugging & Monitoring with Pydantic Logfire](https://ai.pydantic.dev/logfire/index.md)

- [ ] 14.1 Pydantic Logfire 集成
  - 什么是 Logfire？
  - 安装和配置
  - 自动追踪 Agent 运行

- [ ] 14.2 OpenTelemetry 集成
  - `instrument` 参数
  - `instrument_all()` 全局追踪
  - 自定义 span 和属性

- [ ] 14.3 监控指标
  - Token 使用追踪
  - 延迟监控
  - 错误率统计
  - 成本追踪

- [ ] 14.4 实践：构建监控仪表板
  - 实时监控面板
  - 告警配置
  - 性能分析

**输出：** 一个带有完整监控的 Agent 系统

---

## Part 3: 高级篇 - 复杂 Agent 系统

### Chapter 15: MCP 集成 - 连接外部工具生态
**项目：MCP 驱动的全能助手**

> 官方文档：[MCP Overview](https://ai.pydantic.dev/mcp/overview/index.md) | [MCP Client](https://ai.pydantic.dev/mcp/client/index.md) | [MCP Server](https://ai.pydantic.dev/mcp/server/index.md)

- [ ] 15.1 MCP 基础
  - 什么是 Model Context Protocol？
  - MCP Server vs MCP Client
  - 为什么 MCP 重要？

- [ ] 15.2 使用 MCPServer 连接工具
  - `MCPServer` 类：基于 MCP SDK
  - 配置远程和本地 MCP 服务器
  - 工具发现和调用

- [ ] 15.3 使用 FastMCP Client
  - FastMCP 客户端的优势
  - 工具转换（Tool Transformation）
  - OAuth 配置简化

- [ ] 15.4 构建 MCP Server
  - 将 Agent 暴露为 MCP Server
  - 工具注册和描述
  - 安全性考虑

- [ ] 15.5 实践：构建 MCP 驱动的开发助手
  - 连接 GitHub MCP Server
  - 连接数据库 MCP Server
  - 多 MCP 服务器协作

**输出：** 一个能够访问多种外部工具的全能 Agent

---

### Chapter 16: 第三方工具集成
**项目：扩展工具生态**

> 官方文档：[Third-Party Tools](https://ai.pydantic.dev/third-party-tools/index.md)

- [ ] 16.1 第三方工具概述
  - 社区工具生态
  - 工具质量评估

- [ ] 16.2 集成常用第三方工具
  - 搜索引擎工具
  - 数据库工具
  - API 调用工具

- [ ] 16.3 自定义工具封装
  - 将外部 API 封装为工具
  - 工具版本管理
  - 错误处理和重试

- [ ] 16.4 实践：构建工具市场
  - 工具发现和安装
  - 工具配置管理
  - 使用统计

**输出：** 一个可扩展的工具集成系统

---

### Chapter 17: Toolsets - 工具集管理
**项目：动态工具管理系统**

> 官方文档：[Toolsets](https://ai.pydantic.dev/toolsets/index.md) | [Deferred Tools](https://ai.pydantic.dev/deferred-tools/index.md)

- [ ] 17.1 工具集概念
  - `AbstractToolset` 抽象类
  - `FunctionToolset`：函数工具集
  - 动态工具注册

- [ ] 17.2 工具集操作
  - `get_tools()` 和 `call_tool()`
  - `PreparedToolset`：工具预处理
  - `RenamedToolset`：工具重命名

- [ ] 17.3 延迟工具（Deferred Tools）
  - 什么是延迟工具？
  - `DeferredToolRequests` 和 `DeferredToolResults`
  - 跨请求的工具执行
  - 前后端分离场景

- [ ] 17.4 实践：构建可配置的工具系统
  - 根据用户角色启用不同工具
  - 运行时动态添加工具
  - 工具使用统计

**输出：** 一个灵活的动态工具管理系统

---

### Chapter 18: Embeddings - 向量嵌入
**项目：语义搜索系统**

> 官方文档：[Embeddings](https://ai.pydantic.dev/embeddings/index.md)

- [ ] 18.1 Embeddings 基础
  - 什么是向量嵌入？
  - 嵌入模型选择
  - 维度和相似度

- [ ] 18.2 PydanticAI Embeddings API
  - 支持的嵌入模型
  - 批量嵌入
  - 异步处理

- [ ] 18.3 向量存储集成
  - Faiss / Milvus / Qdrant
  - 索引和检索
  - 相似度搜索

- [ ] 18.4 实践：构建语义搜索引擎
  - 文档向量化
  - 相似度排序
  - 混合搜索策略

**输出：** 一个高效的语义搜索系统

---

### Chapter 19: Pydantic Graph - 工作流图
**项目：复杂工作流编排系统**

> 官方文档：[Graph Overview](https://ai.pydantic.dev/graph/index.md)

- [ ] 19.1 Pydantic Graph 概述
  - 为什么需要 Graph？
  - Graph vs 简单 Agent
  - 适用场景

- [ ] 19.2 Graph 核心概念
  - `Node` - 节点定义
  - `Step` - 执行步骤
  - `Decision` - 决策节点
  - `Join` - 汇聚节点

- [ ] 19.3 构建工作流图
  - `GraphBuilder` 使用
  - 节点连接和流程控制
  - 条件分支和循环

- [ ] 19.4 Graph 持久化
  - 状态序列化
  - 断点恢复
  - 持久化后端

- [ ] 19.5 可视化
  - Mermaid 图表生成
  - 运行时状态展示

- [ ] 19.6 实践：构建审批工作流
  - 多级审批流程
  - 并行处理
  - 异常回滚

**输出：** 一个可视化的复杂工作流系统

---

### Chapter 20: 多 Agent 协作与 A2A
**项目：复杂任务的 Agent 团队**

> 官方文档：[Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/index.md) | [Agent2Agent (A2A)](https://ai.pydantic.dev/a2a/index.md)

- [ ] 20.1 为什么需要多 Agent？
  - 单 Agent 的局限
  - 任务分解

- [ ] 20.2 多 Agent 模式
  - 委托模式（Delegation）
  - 管道模式（Pipeline）
  - 协作模式（Collaboration）

- [ ] 20.3 Agent2Agent (A2A) 协议
  - 什么是 A2A？
  - Agent 间的通信机制
  - 标准化的消息格式
  - `fasta2a` 模块

- [ ] 20.4 设计 Agent 团队
  - 专家 Agent
  - 协调 Agent
  - 通信机制

- [ ] 20.5 实践：数据分析团队
  - 数据获取 Agent
  - 分析 Agent
  - 报告生成 Agent

**输出：** 一个多 Agent 协作的数据分析系统

---

### Chapter 21: Durable Execution - 持久化执行
**项目：可恢复的长任务 Agent**

> 官方文档：[Durable Execution Overview](https://ai.pydantic.dev/durable_execution/overview/index.md) | [DBOS](https://ai.pydantic.dev/durable_execution/dbos/index.md) | [Prefect](https://ai.pydantic.dev/durable_execution/prefect/index.md) | [Temporal](https://ai.pydantic.dev/durable_execution/temporal/index.md)

- [ ] 21.1 为什么需要持久化执行？
  - 长时间运行的任务
  - 网络中断和 API 失败
  - 需要人工干预的工作流

- [ ] 21.2 持久化执行框架集成
  - **DBOS** - 数据库驱动的持久化
  - **Prefect** - 工作流编排平台
  - **Temporal** - 分布式工作流引擎

- [ ] 21.3 实现 Durable Agent
  - 状态序列化和恢复
  - 检查点机制
  - 幂等性设计

- [ ] 21.4 错误恢复
  - 自动重试策略
  - 部分结果保存
  - 回滚机制

- [ ] 21.5 实践：构建数据迁移助手
  - 大规模数据处理
  - 断点续传
  - 进度报告

**输出：** 一个能够从中断中恢复的可靠 Agent

---

### Chapter 22: UI Event Streams - 前端集成
**项目：实时交互的 Web Agent**

> 官方文档：[UI Overview](https://ai.pydantic.dev/ui/overview/index.md) | [AG-UI](https://ai.pydantic.dev/ui/ag-ui/index.md) | [Vercel AI](https://ai.pydantic.dev/ui/vercel-ai/index.md)

- [ ] 22.1 UI Event Streams 概述
  - 为什么需要标准化事件流？
  - 前后端通信模式
  - 支持的协议

- [ ] 22.2 AG-UI 集成
  - AG-UI 协议介绍
  - 事件类型和格式
  - 前端组件集成

- [ ] 22.3 Vercel AI SDK 集成
  - Vercel AI SDK 简介
  - 流式响应处理
  - React/Next.js 组件

- [ ] 22.4 实践：构建 Chat 应用
  - 实时消息流
  - 工具调用可视化
  - 打字机效果

**输出：** 一个实时交互的 Web Chat 应用

---

### Chapter 23: Pydantic Evals - Agent 评估
**项目：Agent 性能评估系统**

> 官方文档：[Evals Overview](https://ai.pydantic.dev/evals/index.md)

- [ ] 23.1 为什么需要评估？
  - LLM 输出的不确定性
  - 质量保证
  - 持续改进

- [ ] 23.2 Pydantic Evals 框架
  - `Dataset` - 测试数据集
  - `Evaluator` - 评估器
  - `Generation` - 生成和评估

- [ ] 23.3 评估指标
  - 准确性评估
  - 相关性评估
  - 安全性评估
  - 自定义评估器

- [ ] 23.4 评估报告
  - 报告生成
  - 可视化分析
  - 趋势追踪

- [ ] 23.5 OpenTelemetry 集成
  - 评估数据追踪
  - 与监控系统集成

- [ ] 23.6 实践：构建自动化评估流水线
  - CI/CD 集成
  - 回归测试
  - A/B 测试

**输出：** 一个完整的 Agent 评估系统

---

### Chapter 24: Text2SQL - 智能问数系统
**项目：自然语言查询数据库**

> 官方示例：[SQL Generation](https://ai.pydantic.dev/examples/sql-gen/index.md)

- [ ] 24.1 Text2SQL 的挑战
  - Schema 理解
  - SQL 生成
  - 结果解释

- [ ] 24.2 设计 Text2SQL Agent
  - Schema 检索工具
  - SQL 生成和验证
  - 结果格式化

- [ ] 24.3 优化 SQL 生成质量
  - Few-shot 示例
  - Schema 描述优化
  - 复杂查询的分解

- [ ] 24.4 安全执行 SQL
  - 只读限制
  - 查询超时
  - 用户确认机制（Human-in-the-Loop）

- [ ] 24.5 结果解释和可视化
  - 自然语言描述结果
  - 生成图表
  - 多表关联查询

**输出：** 一个完整的 Text2SQL 智能问数系统

---

### Chapter 25: RAG - 为 Agent 赋予领域知识
**项目：企业知识库问答系统**

> 官方示例：[RAG](https://ai.pydantic.dev/examples/rag/index.md)

- [ ] 25.1 RAG 基础
  - 什么是 RAG？
  - 与 Embeddings 的关系
  - 架构设计

- [ ] 25.2 构建知识库
  - 文档分块策略
  - 向量化和索引
  - 元数据管理

- [ ] 25.3 实现 RAG Agent
  - 检索工具
  - 重排序
  - 引用溯源

- [ ] 25.4 优化检索质量
  - Hybrid Search
  - Query 改写
  - 多跳推理

**输出：** 一个基于 RAG 的企业知识问答 Agent

---

### Chapter 26: Agent 的提示工程
**项目：优化 Agent 性能**

- [ ] 26.1 Instructions 设计
  - 静态 `instructions` vs 动态生成
  - `@agent.instructions` 装饰器
  - 角色定义和行为规范

- [ ] 26.2 工具描述优化
  - 清晰的函数名
  - 详细的文档字符串
  - 参数说明
  - `Tool.from_schema()` 自定义 schema

- [ ] 26.3 Few-shot Learning
  - 添加示例
  - 示例选择策略
  - 动态示例

- [ ] 26.4 提示迭代和评估
  - 使用 Pydantic Evals
  - A/B 测试
  - 持续优化

**输出：** 一套系统的 Prompt 优化方法论

---

### Chapter 27: 部署和生产化
**项目：将 Agent 部署到生产环境**

> 官方示例：[Chat App with FastAPI](https://ai.pydantic.dev/examples/chat-app/index.md)

- [ ] 27.1 API 封装
  - FastAPI 集成
  - RESTful 接口设计
  - WebSocket 支持（流式）
  - UI 事件流标准集成

- [ ] 27.2 并发和性能
  - 异步处理（PydanticAI 原生异步）
  - 连接池管理
  - 缓存策略

- [ ] 27.3 监控和可观测性
  - Logfire / OpenTelemetry 集成
  - 日志系统
  - 指标收集
  - 分布式追踪

- [ ] 27.4 成本优化
  - Token 使用优化
  - 缓存重复查询
  - 模型选择策略

- [ ] 27.5 安全性
  - 认证和授权
  - 数据脱敏
  - 审计日志

**输出：** 一个生产就绪的 Agent 服务

---

## Part 4: 实战篇 - 综合项目

### Chapter 28: 综合项目 1 - 游戏数据分析助手
**为你的游戏数据分析工作定制**

> 参考示例：[Data Analyst](https://ai.pydantic.dev/examples/data-analyst/index.md)

- [ ] 28.1 需求分析
  - 常见的分析任务
  - 用户画像
  - 功能优先级

- [ ] 28.2 系统设计
  - Agent 架构
  - 工具设计
  - 数据流

- [ ] 28.3 核心功能实现
  - 多游戏支持
  - 复杂报表生成
  - 趋势分析和预测

- [ ] 28.4 优化和迭代
  - 根据使用反馈改进
  - 性能优化
  - 新功能添加

**输出：** 一个为游戏数据分析定制的生产级 Agent 系统

---

### Chapter 29: 综合项目 2 - 自定义项目
**根据你的兴趣选择**

可选方向（参考官方示例）：
- [Bank Support](https://ai.pydantic.dev/examples/bank-support/index.md) - 银行客服助手
- [Flight Booking](https://ai.pydantic.dev/examples/flight-booking/index.md) - 机票预订助手
- [Weather Agent](https://ai.pydantic.dev/examples/weather-agent/index.md) - 天气查询助手
- [Slack Lead Qualifier](https://ai.pydantic.dev/examples/slack-lead-qualifier/index.md) - Slack 销售线索筛选
- 代码审查和重构助手
- 技术文档生成器
- DevOps 运维助手

---

## Part 5: 命令行工具

### Chapter 30: Clai - PydanticAI CLI
**快速原型和测试**

> 官方文档：[Clai](https://ai.pydantic.dev/cli/index.md)

- [ ] 30.1 Clai 简介
  - 安装和配置
  - 基本用法

- [ ] 30.2 常用命令
  - 快速测试 Agent
  - 调试工具调用
  - 性能分析

- [ ] 30.3 实践：使用 Clai 加速开发
  - 快速原型验证
  - 交互式调试
  - 批量测试

**输出：** 熟练使用 Clai 进行开发和调试

---

## 附录

### Appendix A: PydanticAI API 参考

- **Agent 类完整参考**
  - 构造参数：`model`、`output_type`、`instructions`、`deps_type`、`tools`、`builtin_tools`、`toolsets` 等
  - 运行方法：`run()`、`run_sync()`、`run_stream()`
  - 装饰器：`@agent.tool`、`@agent.tool_plain`、`@agent.output`、`@agent.instructions`

- **工具装饰器详解**
  - `@agent.tool`：需要 `RunContext` 的工具
  - `@agent.tool_plain`：不需要上下文的简单工具
  - `Tool` 类：精细控制工具定义

- **配置选项**
  - `ModelSettings`：模型请求设置
  - `InstrumentationSettings`：追踪设置

- **常用模式和代码片段**

### Appendix B: API 迁移指南（重要）

从旧版本迁移时注意以下变化：

| 旧 API | 新 API | 说明 |
|--------|--------|------|
| `result_type` | `output_type` | Agent 构造参数 |
| `system_prompt` | `instructions` | 推荐使用新参数 |
| `result.data` | `result.output` | 获取运行结果 |
| `result.all_messages_json()` | `result.all_messages()` | 获取消息历史 |

### Appendix C: 最佳实践清单

- Agent 设计检查清单
- 安全检查清单
- 性能优化清单
- 测试覆盖清单

### Appendix D: 故障排查指南

> 官方文档：[Troubleshooting](https://ai.pydantic.dev/troubleshooting/index.md) | [Getting Help](https://ai.pydantic.dev/help/index.md)

- 常见错误和解决方案
- 调试技巧
- 社区资源

### Appendix E: 从其他框架迁移

> 官方文档：[Upgrade Guide](https://ai.pydantic.dev/changelog/index.md)

- LangChain → PydanticAI
- LlamaIndex → PydanticAI
- 对比和选型指南

---

## 学习建议

1. **按顺序学习**：Part 1 和 Part 2 建议按顺序完成，Part 3 和 Part 4 可根据需求选择

2. **动手实践**：每个章节都要完成实际项目，不要只看不做

3. **代码复用**：保留每个项目的代码，后续项目可以复用前面的成果

4. **持续迭代**：项目完成后继续改进，添加新功能

5. **记录笔记**：在 `learning_notes.md` 中记录你的心得和踩坑经验

6. **提问和讨论**：遇到问题随时提问，学习是双向的过程

7. **关注官方文档**：PydanticAI 更新较快，建议定期查看 https://ai.pydantic.dev/

8. **查阅 llms.txt**：https://ai.pydantic.dev/llms.txt 提供了结构化的文档索引

---

## 预计学习时间

- **Part 1 基础篇**（Chapter 1-6）：2-3 周（每天 2-3 小时）
- **Part 2 进阶篇**（Chapter 7-14）：3-4 周（每天 2-3 小时）
- **Part 3 高级篇**（Chapter 15-27）：5-6 周（每天 2-3 小时）
- **Part 4 实战篇**（Chapter 28-29）：2-4 周（根据项目复杂度）
- **Part 5 工具篇**（Chapter 30）：1-2 天

**总计**：12-17 周可以完成从入门到精通的学习路径

---

## 更新日志

- **2025-01**: 课程大纲审核和更新（第二版）
  - 参考官方 llms.txt 完整重构课程结构
  - 新增 Chapter 5：多模态输入（Image, Audio, Video & Document）
  - 新增 Chapter 9：Thinking 模型思考过程
  - 新增 Chapter 10：Direct Model Requests 直接模型请求
  - 新增 Chapter 16：Third-Party Tools 第三方工具
  - 新增 Chapter 18：Embeddings 向量嵌入
  - 新增 Chapter 19：Pydantic Graph 工作流图
  - 新增 Chapter 22：UI Event Streams（AG-UI、Vercel AI）
  - 新增 Chapter 23：Pydantic Evals Agent 评估
  - 新增 Chapter 30：Clai 命令行工具
  - 补充 Durable Execution 的具体框架集成（DBOS、Prefect、Temporal）
  - 为每个章节添加官方文档链接引用
  - 调整章节顺序，确保学习路径更合理
