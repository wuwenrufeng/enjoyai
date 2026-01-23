# PydanticAI 核心概念学习总结

## 1. Agent 的执行流程（ReAct 模式）

### 完整的执行流程

当调用 `await agent.run(user_input, deps=config, message_history=history)` 时：

```
用户输入 (Question)
    ↓
Agent 组装请求发送给 LLM:
  - 用户当前输入
  - 可用工具的描述
  - System Prompt
  - 历史对话（message_history）
  - LLM 模型参数
    ↓
LLM 思考 (Thought)
    ↓
LLM 决定采取行动 (Action)
  - 调用某个工具
  - 或直接回答
    ↓
Agent 执行工具调用 (Pause & Execute)
    ↓
工具返回结果 (Observation)
    ↓
Agent 将结果发回 LLM
    ↓
LLM 再次决策:
  - 继续调用其他工具？
  - 还是总结输出答案？
    ↓
[可能循环多次]
    ↓
最终答案 (Answer)
```

### ReAct 模式

这个过程被称为 **ReAct（Reasoning and Acting）**：

- **Re**asoning：LLM 推理思考
- **Act**ing：执行实际的工具调用
- 循环往复，直到得出最终答案

## 2. Message History - 对话记忆机制

### 为什么需要 message_history？

```python
history = []
result = await agent.run(user_input, deps=config, message_history=history)
history.extend(result.new_messages())
```

### 核心概念

- **`result.new_messages()`**: 返回本轮对话产生的所有新消息
  - 包括：用户输入、工具调用、工具结果、LLM 回复
  - 不包括：之前对话的历史消息

- **为什么要加到 history？**
  - LLM 本身是**无状态**的，每次调用都是独立的
  - 通过传入 `message_history`，让 LLM 能够"记住"之前的对话
  - 这是一种**基于内存的短期记忆**机制

### 示例对比

**没有 message_history：**

```
用户: "c694bb8a65706484 数据库有哪些表？"
Agent: 返回表列表

用户: "第一个表有哪些字段？"
Agent: ❌ 不知道"第一个表"指的是什么
```

**有 message_history：**

```
用户: "c694bb8a65706484 数据库有哪些表？"
Agent: 返回表列表

用户: "第一个表有哪些字段？"
Agent: ✅ 知道之前返回的表列表，理解"第一个表"的含义
```

## 3. 依赖注入（Dependency Injection）

### 为什么不用全局变量？

**全局变量方式：**

```python
# ❌ 不推荐
client = clickhouse_connect.get_client(...)

@agent.tool
async def list_databases():
    result = client.query("SHOW DATABASES")
    ...
```

**依赖注入方式：**

```python
# ✅ 推荐
@dataclass
class Config:
    ck_client: AsyncClient

agent = Agent(
    model='...',
    deps_type=Config,  # 声明依赖类型
)

@agent.tool
async def list_databases(ctx: RunContext[Config]):
    result = ctx.deps.ck_client.query("SHOW DATABASES")
    ...

# 运行时传入实际的依赖
config = Config(ck_client=actual_client)
result = await agent.run(query, deps=config)
```

### 依赖注入的三大优势

#### 1. 降低耦合性

- 工具函数不依赖全局变量
- 可以在不同上下文中使用相同的工具

#### 2. 可测试性

**场景：** 想测试 `list_databases` 工具是否正确解析返回结果

**全局变量方式：**

```python
# 难以 mock，需要修改全局状态
def test_list_databases():
    global client  # 😱 需要替换全局变量
    client = MockClient()
    # 测试...
```

**依赖注入方式：**

```python
# 轻松 mock
def test_list_databases():
    mock_client = MockClient()
    mock_config = Config(ck_client=mock_client)
    ctx = create_test_context(deps=mock_config)
    result = await list_databases(ctx)
    # 断言测试...
```

#### 3. 灵活性

- 可以在不同环境使用不同的依赖
- 例如：开发环境用本地数据库，生产环境用远程数据库
- 只需传入不同的 `Config` 实例

## 4. 工具注册机制

### @agent.tool 装饰器的作用

```python
@db_agent.tool
async def list_databases(ctx: RunContext[Config]) -> List[str]:
    """列出所有数据库"""
    ...
```

装饰器做了什么：

1. **注册工具**：将函数注册到 Agent 的工具列表
2. **生成描述**：从函数签名和 docstring 自动生成工具描述
3. **类型验证**：利用类型注解验证参数和返回值

### LLM 如何知道可用的工具？

Agent 在发送请求给 LLM 时，会包含所有工具的描述：

```json
{
  "tools": [
    {
      "name": "list_databases",
      "description": "列出所有数据库",
      "parameters": {...}
    },
    {
      "name": "list_tables",
      "description": "列出指定数据库中的所有表",
      "parameters": {
        "database": {"type": "string"}
      }
    }
  ]
}
```

LLM 根据这些描述，决定调用哪个工具。

## 5. 参数化查询语法

### ClickHouse 参数化查询格式

```python
# ✅ 正确
query = "SELECT name FROM system.tables WHERE database = %(db)s"
parameters = {"db": "my_database"}

# ❌ 错误 - 缺少类型标识符
query = "SELECT name FROM system.tables WHERE database = %(db)"
```

### 类型标识符

- `%(name)s` - 字符串 (string)
- `%(count)d` - 整数 (decimal)
- `%(price)f` - 浮点数 (float)

### 为什么需要类型标识符？

Python 的 `%` 格式化需要知道如何格式化值：

```python
"Hello %(name)s" % {"name": "World"}  # ✅
"Hello %(name)" % {"name": "World"}   # ❌ ValueError: incomplete format
```

## 6. 工具设计的权衡

### 细粒度工具 vs 通用工具

**细粒度工具（当前实现）：**

```python
@agent.tool
async def list_databases(ctx): ...

@agent.tool
async def list_tables(ctx, database: str): ...

@agent.tool
async def describe_table(ctx, database: str, table: str): ...
```

**优点：**

- ✅ 安全，不会有 SQL 注入
- ✅ 可控，明确知道能做什么
- ✅ 容易测试和维护

**缺点：**

- ❌ 可能需要无限添加工具
- ❌ 无法应对所有查询需求

**通用工具（可能的实现）：**

```python
@agent.tool
async def run_select_query(ctx, sql: str): ...
```

**优点：**

- ✅ 灵活，一个工具应对所有需求
- ✅ 不需要为每个查询写工具

**缺点：**

- ❌ SQL 注入风险
- ❌ 难以控制和审计
- ❌ LLM 可能生成错误的 SQL

### 实践建议

平衡安全性和灵活性：

1. 常用的、安全的操作 → 细粒度工具
2. 需要灵活性的查询 → 通用工具 + 限制条件：
   - 只允许 SELECT 查询
   - 只允许查询特定的表（如 system.\*）
   - 需要用户确认后再执行

## 总结

这个 ClickHouse 表结构助手项目教会我们：

1. **Agent = LLM + Tools + 编排逻辑**
   - LLM 是大脑（思考和决策）
   - Tools 是手（执行具体操作）
   - ReAct 模式是神经系统（协调思考和行动）

2. **核心设计模式**
   - 依赖注入：提高可测试性和灵活性
   - 对话记忆：让 Agent 能够保持上下文
   - 工具注册：声明式定义 Agent 的能力

3. **实践经验**
   - 参数化查询要加类型标识符
   - 工具设计要在安全和灵活之间权衡
   - 理解 LLM 的无状态本质很重要
