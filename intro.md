
✅ LangGraph 管流程：控制流程、模块解耦、任务驱动执行
✅ MCP 给上下文：负责多轮记忆管理（Multi-round Context Provider）
✅ GPT-4o 来推理：负责自然语言理解、SQL生成、答案解释


## 工作流分析
该项目是一个基于 LangGraph 和 DeepSeek 大语言模型的 SQL 查询助手，使用 Gradio 作为界面。工作流主要包括以下步骤：

1. 初始化配置 ：加载环境变量，配置 LangSmith（可选），初始化 LLM
2. 数据库设置 ：创建或连接到 SQLite 数据库，设置示例数据
3. Agent 工作流定义 ：使用 LangGraph 创建状态图，定义节点和边
4. 用户交互 ：通过 Gradio 界面接收用户输入，调用 Agent 处理查询
5. 结果展示 ：将 Agent 处理结果以流式方式返回给用户

```mermaid
flowchart TD
    A[开始] --> B[加载环境变量]
    B --> C[配置 LangSmith]
    C --> D[初始化 LLM]
    D --> E[设置数据库]
    E --> F[创建 SQL Agent]
    F --> G[启动 Gradio 界面]
    
    G --> H[用户输入问题]
    H --> I[创建初始状态]
    I --> J[调用 Agent 处理]
    
    J --> K{识别意图}
    K -->|GET_SCHEMA| L[直接返回数据库模式]
    K -->|EXECUTE_QUERY/GET_INFO| M[获取数据库模式]
    
    M --> N[生成 SQL 查询]
    N --> O[执行 SQL 查询]
    O --> P[生成最终回答]
    
    L --> Q[返回结果给用户]
    P --> Q
    
    Q --> R{继续对话?}
    R -->|是| H
    R -->|否| S[结束]
```

## 数据流分析
数据在系统中的流动路径如下：

1. 输入数据 ：用户问题、环境变量配置、对话历史
2. 处理数据 ：意图识别、SQL 查询生成、查询执行结果
3. 输出数据 ：最终回答、中间步骤（思考过程、SQL 查询、查询结果）

```mermaid
flowchart LR
    subgraph 输入
        A1[用户问题]
        A2[环境变量]
        A3[对话历史]
    end
    
    subgraph 处理
        B1[意图识别]
        B2[数据库模式获取]
        B3[SQL查询生成]
        B4[SQL查询执行]
        B5[回答生成]
    end
    
    subgraph 状态
        C1[AgentState]
        C2[question]
        C3[thoughts]
        C4[intent]
        C5[sql_query]
        C6[sql_result]
        C7[answer]
        C8[error]
    end
    
    subgraph 输出
        D1[最终回答]
        D2[思考过程]
        D3[SQL查询]
        D4[查询结果]
    end
    
    A1 --> C1
    A2 --> B1
    A2 --> B3
    A2 --> B4
    A3 --> C1
    
    C1 --> B1
    B1 --> C4
    C4 --> B2
    B2 --> B3
    B3 --> C5
    C5 --> B4
    B4 --> C6
    C6 --> B5
    B5 --> C7
    
    C7 --> D1
    C3 --> D2
    C5 --> D3
    C6 --> D4
```