# PydanticAI Agent 开发学习进度

> 📅 开始日期：2025-01-22  
> 🎯 目标：从入门到精通 PydanticAI Agent 开发  
> 📚 课程大纲：见 [COURSE_OUTLINE.md](./COURSE_OUTLINE.md)  
> 📄 官方文档：https://ai.pydantic.dev/  
> 📄 LLMs.txt：https://ai.pydantic.dev/llms.txt

---

## 当前进度概览

- **当前章节**: Chapter 2 - 工具函数（接近完成）
- **完成度**: Part 1 基础篇 - 30% (1.8/6 章节)
- **学习天数**: 2 天
- **累计实践项目**: 1 个（ClickHouse 表结构助手 - 4 个工具）

### 进度条
```
Part 1 基础篇    [██████░░░░░░░░░░░░░░] 30%  (1.8/6)
Part 2 进阶篇    [░░░░░░░░░░░░░░░░░░░░]  0%  (0/8)
Part 3 高级篇    [░░░░░░░░░░░░░░░░░░░░]  0%  (0/13)
Part 4 实战篇    [░░░░░░░░░░░░░░░░░░░░]  0%  (0/2)
Part 5 工具篇    [░░░░░░░░░░░░░░░░░░░░]  0%  (0/1)
─────────────────────────────────────────────
总体进度         [██░░░░░░░░░░░░░░░░░░]  6%  (1.8/30)
```

---

## Part 1: 基础篇 (Chapter 1-6)

### ✅ Chapter 1: Hello Agent World
**状态**: 已完成  
**完成日期**: 2025-01-22  
**项目**: 简单问答助手（跳过，直接进入实际项目）

> 官方文档：[Agents](https://ai.pydantic.dev/agents/index.md) | [Installation](https://ai.pydantic.dev/install/index.md)

#### 完成的小节
- [x] 1.1 环境搭建
- [x] 1.2 理解 Agent 的核心组件
- [x] 1.3 支持的模型提供商（使用 Kimi）
- [x] 1.4 第一个 Agent 程序

#### 学习笔记
- 使用了 Kimi 模型（`openai:kimi-k2-0711-preview`）而不是 Claude（成本考虑）
- 理解了 Agent 的基本结构
- 掌握了如何运行 Agent

---

### 🔄 Chapter 2: 工具函数 - Agent 的手和脚
**状态**: 进行中 (80%)  
**开始日期**: 2025-01-22  
**项目**: ClickHouse 表结构助手

> 官方文档：[Function Tools](https://ai.pydantic.dev/tools/index.md) | [Dependencies](https://ai.pydantic.dev/dependencies/index.md)

#### 完成的小节
- [x] 2.1 什么是工具函数
  - 理解了 `@agent.tool` 装饰器
  - 学会了如何定义工具函数
  - 理解了工具如何被 LLM 调用

- [x] 2.2 设计工具函数
  - 实现了 `list_database()` - 列出所有数据库
  - 实现了 `list_tables(database)` - 列出指定数据库的表
  - 实现了 `describe_table(database, table)` - 查看表字段信息
  - 学会了使用 Pydantic 模型（`Column`）作为返回值

- [x] 2.3 依赖注入
  - 创建了 `Config` dataclass 管理 ClickHouse 连接
  - 理解了 `deps_type` 和 `RunContext` 的作用
  - 掌握了如何在 `run()` 时传入依赖

#### 进行中的小节
- [ ] 2.4 工具函数的高级模式
  - `Tool` 类精细控制
  - 错误处理：`ModelRetry` 异常
  - 工具超时设置

- [x] 2.5 工具设计的权衡
  - ✅ 已实现 `find_tables_with_column` 工具

#### 当前代码实现
```python
# clickhouse_agent/agent.py 核心结构
@dataclass
class Config:
    ck_client: AsyncClient

db_agent = agent.Agent(
    "openai:kimi-k2-0711-preview",
    system_prompt="你是一个ClickHouse数据库查询助手",
    deps_type=Config,
)

# 已实现的工具（4个）
@db_agent.tool
async def list_database(ctx: RunContext[Config]) -> List[str]: ...

@db_agent.tool
async def list_tables(ctx: RunContext[Config], database: str) -> List[str]: ...

@db_agent.tool
async def describe_table(ctx: RunContext[Config], database: str, table: str) -> List[Column]: ...

@db_agent.tool
async def find_tables_with_column(ctx: RunContext[Config], column_name: str) -> List[Table]: ...
```

#### 关键收获
1. **ReAct 模式**：理解了 Agent 的 Question → Thought → Action → Observation → Answer 循环
2. **依赖注入**：降低耦合性、提高可测试性、增加灵活性
3. **参数化查询**：学会了使用 `%(param)s` 语法防止 SQL 注入
4. **对话记忆**：通过 `message_history` 实现上下文保持

#### 遇到的问题和解决
| 问题 | 解决方案 |
|------|---------|
| `result.result_rows()` vs `result.result_rows` | 发现是属性而不是方法 |
| `%(db)` 报 ValueError: incomplete format | 需要添加类型标识符 `%(db)s` |
| 如何从查询结果提取数据 | 使用 `zip(result.column_names, row)` 创建字典 |

#### 下一步计划
- [x] ~~完成 `find_tables_with_column` 工具的实现~~ ✅ 2025-01-22
- [ ] 学习 `ModelRetry` 异常处理
- [ ] 了解工具超时设置

---

### ⏳ Chapter 3: 高级工具特性
**状态**: 未开始  
**计划**: 待 Chapter 2 完成后开始

> 官方文档：[Advanced Tool Features](https://ai.pydantic.dev/tools-advanced/index.md)

#### 待完成的小节
- [ ] 3.1 工具返回多模态内容（ImageUrl, DocumentUrl）
- [ ] 3.2 ToolReturn - 精细控制返回值
- [ ] 3.3 Tool.from_schema()
- [ ] 3.4 实践：构建文件分析助手

---

### ⏳ Chapter 4: 内置工具与通用工具
**状态**: 未开始

> 官方文档：[Built-in Tools](https://ai.pydantic.dev/builtin-tools/index.md) | [Common Tools](https://ai.pydantic.dev/common-tools/index.md)

#### 待完成的小节
- [ ] 4.1 WebFetchTool - 网页抓取
- [ ] 4.2 ImageGenerationTool - 图像生成
- [ ] 4.3 MemoryTool - 对话记忆
- [ ] 4.4 CodeExecutionTool - 代码执行
- [ ] 4.5 Common Tools - 通用工具
- [ ] 4.6 实践：构建多功能研究助手

---

### ⏳ Chapter 5: 多模态输入
**状态**: 未开始

> 官方文档：[Image, Audio, Video & Document Input](https://ai.pydantic.dev/input/index.md)

#### 待完成的小节
- [ ] 5.1 图片输入
- [ ] 5.2 音频输入
- [ ] 5.3 视频输入
- [ ] 5.4 文档输入
- [ ] 5.5 实践：构建多模态分析助手

---

### ⏳ Chapter 6: 对话管理与消息历史
**状态**: 部分了解（通过实践）

> 官方文档：[Messages and chat history](https://ai.pydantic.dev/message-history/index.md)

#### 待完成的小节
- [x] 6.1 理解 LLM 的无状态本质（已通过实践了解）
- [ ] 6.2 实现对话记忆（已基本实现，需深入学习）
- [ ] 6.3 上下文引用
- [ ] 6.4 多轮对话的设计模式

#### 当前实现
```python
# 已在 agent.py 中实现基本对话记忆
history = []
while True:
    result = await db_agent.run(user_input, deps=config, message_history=history)
    history.extend(result.new_messages())
```

---

## Part 2: 进阶篇 (Chapter 7-14)

### ⏳ Chapter 7: 结构化输出
**状态**: 未开始
> 官方文档：[Output](https://ai.pydantic.dev/output/index.md)

### ⏳ Chapter 8: 流式响应
**状态**: 未开始

### ⏳ Chapter 9: Thinking - 模型思考过程
**状态**: 未开始
> 官方文档：[Thinking](https://ai.pydantic.dev/thinking/index.md)

### ⏳ Chapter 10: 直接模型请求
**状态**: 未开始
> 官方文档：[Direct Model Requests](https://ai.pydantic.dev/direct/index.md)

### ⏳ Chapter 11: HTTP 重试与错误处理
**状态**: 未开始
> 官方文档：[HTTP Request Retries](https://ai.pydantic.dev/retries/index.md)

### ⏳ Chapter 12: Human-in-the-Loop - 工具审批机制
**状态**: 未开始

### ⏳ Chapter 13: Agent 测试
**状态**: 未开始
> 官方文档：[Testing](https://ai.pydantic.dev/testing/index.md)

### ⏳ Chapter 14: 调试与监控
**状态**: 未开始
> 官方文档：[Debugging & Monitoring with Pydantic Logfire](https://ai.pydantic.dev/logfire/index.md)

---

## Part 3: 高级篇 (Chapter 15-27)

### ⏳ Chapter 15: MCP 集成
**状态**: 未开始
> 官方文档：[MCP Overview](https://ai.pydantic.dev/mcp/overview/index.md)

### ⏳ Chapter 16: 第三方工具集成
**状态**: 未开始
> 官方文档：[Third-Party Tools](https://ai.pydantic.dev/third-party-tools/index.md)

### ⏳ Chapter 17: Toolsets - 工具集管理
**状态**: 未开始
> 官方文档：[Toolsets](https://ai.pydantic.dev/toolsets/index.md)

### ⏳ Chapter 18: Embeddings - 向量嵌入
**状态**: 未开始
> 官方文档：[Embeddings](https://ai.pydantic.dev/embeddings/index.md)

### ⏳ Chapter 19: Pydantic Graph - 工作流图
**状态**: 未开始
> 官方文档：[Graph Overview](https://ai.pydantic.dev/graph/index.md)

### ⏳ Chapter 20: 多 Agent 协作与 A2A
**状态**: 未开始
> 官方文档：[Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/index.md) | [A2A](https://ai.pydantic.dev/a2a/index.md)

### ⏳ Chapter 21: Durable Execution - 持久化执行
**状态**: 未开始
> 官方文档：[Durable Execution](https://ai.pydantic.dev/durable_execution/overview/index.md)

### ⏳ Chapter 22: UI Event Streams - 前端集成
**状态**: 未开始
> 官方文档：[UI Overview](https://ai.pydantic.dev/ui/overview/index.md)

### ⏳ Chapter 23: Pydantic Evals - Agent 评估
**状态**: 未开始
> 官方文档：[Evals Overview](https://ai.pydantic.dev/evals/index.md)

### ⏳ Chapter 24: Text2SQL - 智能问数系统 ⭐
**状态**: 未开始  
**备注**: 🎯 **核心目标项目** - 为游戏数据分析工作定制
> 官方示例：[SQL Generation](https://ai.pydantic.dev/examples/sql-gen/index.md)

### ⏳ Chapter 25: RAG - 领域知识
**状态**: 未开始
> 官方示例：[RAG](https://ai.pydantic.dev/examples/rag/index.md)

### ⏳ Chapter 26: Agent 的提示工程
**状态**: 未开始

### ⏳ Chapter 27: 部署和生产化
**状态**: 未开始
> 官方示例：[Chat App with FastAPI](https://ai.pydantic.dev/examples/chat-app/index.md)

---

## Part 4: 实战篇 (Chapter 28-29)

### ⏳ Chapter 28: 综合项目 1 - 游戏数据分析助手 ⭐
**状态**: 未开始  
**备注**: 🎯 **最终目标** - 结合 Text2SQL 构建完整系统
> 参考示例：[Data Analyst](https://ai.pydantic.dev/examples/data-analyst/index.md)

### ⏳ Chapter 29: 综合项目 2 - 自定义项目
**状态**: 未开始

---

## Part 5: 工具篇 (Chapter 30)

### ⏳ Chapter 30: Clai - PydanticAI CLI
**状态**: 未开始
> 官方文档：[Clai](https://ai.pydantic.dev/cli/index.md)

---

## 学习统计

### 时间投入
| 日期 | 学习时长 | 内容 |
|------|---------|------|
| 2025-01-22 | 4 小时 | Chapter 1-2，ClickHouse Agent 基础 |
| 2025-01-22 | 0.5 小时 | 实现 find_tables_with_column 工具 |
| **总计** | **4.5 小时** | |

### 技能掌握度
| 技能点 | 掌握程度 | 说明 |
|--------|----------|------|
| Agent 基础概念 | ⭐⭐⭐⭐⭐ | 完全理解 |
| 工具函数定义 (`@agent.tool`) | ⭐⭐⭐⭐⭐ | 完全掌握 |
| 依赖注入 (`deps_type`, `RunContext`) | ⭐⭐⭐⭐☆ | 理解原理，需要更多实践 |
| 对话记忆 (`message_history`) | ⭐⭐⭐☆☆ | 基本理解，未深入实践 |
| 异步编程 | ⭐⭐⭐⭐☆ | 基本熟悉 |
| ClickHouse 查询 | ⭐⭐⭐⭐⭐ | 工作中常用 |
| Pydantic 模型 | ⭐⭐⭐⭐☆ | 理解基础，需要更多高级用法 |
| 结构化输出 (`output_type`) | ⭐☆☆☆☆ | 待学习 |
| 流式响应 | ⭐☆☆☆☆ | 待学习 |
| MCP 集成 | ⭐☆☆☆☆ | 待学习 |
| 测试 (`TestModel`) | ⭐☆☆☆☆ | 待学习 |

### 项目成果
| 项目 | 状态 | 说明 |
|------|------|------|
| ClickHouse 表结构助手 | 🔄 进行中 | **4 个工具已实现**，支持多轮对话 |
| Text2SQL 智能问数 | ⏳ 待开始 | 核心目标项目 |
| 游戏数据分析助手 | ⏳ 待开始 | 最终目标项目 |

---

## API 变更提醒 ⚠️

学习过程中需注意 PydanticAI 的 API 变更：

| 旧 API | 新 API | 你的代码状态 |
|--------|--------|-------------|
| `system_prompt` | `instructions` | ✅ 已更新 |
| `result_type` | `output_type` | ✅ 未使用 |
| `result.data` | `result.output` | ✅ 已使用新 API |

**建议**：将 `agent.py` 中的 `system_prompt` 改为 `instructions`

---

## 待办事项

### 🔥 立即要做
- [ ] 完成 Chapter 2.4 剩余内容（ModelRetry、工具超时）
- [x] ~~实现 `find_tables_with_column` 工具~~ ✅
- [x] ~~将 `system_prompt` 更新为 `instructions`~~ ✅

### 📅 本周目标
- [ ] 完成 Chapter 2 的所有小节
- [ ] 开始 Chapter 3 高级工具特性
- [ ] 为现有代码添加错误处理

### 📅 短期目标（2 周内）
- [ ] 完成 Part 1 基础篇 (Chapter 1-6)
- [ ] 为 ClickHouse 助手添加更多实用工具
- [ ] 学习 PydanticAI 的测试方法

### 📅 中期目标（1 个月内）
- [ ] 完成 Part 2 进阶篇 (Chapter 7-14)
- [ ] 开始构建 Text2SQL 原型 (Chapter 24)
- [ ] 深入理解 RAG 集成 (Chapter 25)

### 📅 长期目标（3 个月内）
- [ ] 完成整个课程 (30 章)
- [ ] 构建完整的游戏数据分析智能问数系统
- [ ] 将 Agent 部署到生产环境

---

## 学习心得

### 2025-01-22
#### 今日亮点
- ✅ 成功构建了第一个实用的 Agent 系统
- ✅ 深刻理解了 ReAct 模式的工作原理
- ✅ 掌握了依赖注入的优势
- ✅ 理解了 LLM 的无状态本质和对话记忆的必要性

#### 今日挑战
- ⚠️ 最开始对参数化查询的语法不熟悉
- ⚠️ 对 `result.result_rows` 是属性还是方法有疑惑

#### 改进方向
- 学习过程从"随意"变为"结构化"
- 创建了完整的课程大纲和进度追踪
- 需要更加主动地实践，少依赖直接给答案

---

## 资源和参考

### 官方资源
- [PydanticAI 官方文档](https://ai.pydantic.dev/)
- [PydanticAI GitHub](https://github.com/pydantic/pydantic-ai)
- [LLMs.txt](https://ai.pydantic.dev/llms.txt) - 结构化文档索引

### 官方示例（重点关注）
- [SQL Generation](https://ai.pydantic.dev/examples/sql-gen/index.md) - Text2SQL 参考
- [Data Analyst](https://ai.pydantic.dev/examples/data-analyst/index.md) - 数据分析参考
- [RAG](https://ai.pydantic.dev/examples/rag/index.md) - RAG 参考
- [Chat App](https://ai.pydantic.dev/examples/chat-app/index.md) - 部署参考

### 本地文件
- 课程大纲：`COURSE_OUTLINE.md`
- 学习笔记：`learning_notes.md`
- 项目代码：`clickhouse_agent/agent.py`

### 工具和环境
- Python 3.13
- PydanticAI（建议定期更新）
- ClickHouse (生产环境)
- Kimi AI (月之暗面 API)

---

**最后更新**: 2025-01-22  
**下次学习计划**: 完成 Chapter 2.4 剩余内容（ModelRetry、工具超时），或进入 Chapter 3
