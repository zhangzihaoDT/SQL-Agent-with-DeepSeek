


          
# SQL Agent with DeepSeek LLM

## 项目概述

这是一个基于 LangChain 和 DeepSeek 大语言模型构建的 SQL 智能代理系统，通过 Gradio 提供用户友好的界面。该系统能够理解自然语言问题，将其转换为 SQL 查询，执行查询并以易于理解的方式呈现结果。

## 主要特点

- **自然语言转 SQL**：将用户的自然语言问题转换为精确的 SQL 查询
- **意图识别**：智能识别用户查询意图，选择合适的处理流程
- **交互式界面**：使用 Gradio 构建直观的用户界面
- **DeepSeek 集成**：通过火山方舟 (VolcEngine) API 集成 DeepSeek 大语言模型
- **LangSmith 监控**：可选的 LangSmith 集成，用于追踪和优化 LLM 调用

## 目录结构

```
Langchain_chatwithdata/W20方向/
├── .env                # 环境变量配置文件 (需手动创建)
├── sql_agent_app.py    # 主要的应用脚本
├── requirements.txt    # Python 依赖包
├── chinook_agent.db    # SQLite 数据库文件 (应用运行时自动创建)
└── README.md           # 本文档
```

## 技术栈

- **LangChain**：用于构建 LLM 应用的框架
- **LangGraph**：用于构建基于状态的 AI 工作流
- **DeepSeek**：通过火山方舟 API 访问的大语言模型
- **Gradio**：用于创建交互式 Web 界面
- **SQLite**：轻量级数据库
- **LangSmith**：(可选) 用于追踪和监控 LLM 调用

## 环境设置与安装

1. **克隆仓库 (如果适用)**

    ```bash
    # git clone <repository_url>
    # cd Langchain_chatwithdata/W20方向
    ```

2. **创建并激活 Python 虚拟环境 (推荐)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

3. **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

4. **配置环境变量**

    在项目根目录创建一个名为 `.env` 的文件，并填入以下内容:

    ```env
    # VolcEngine Ark API 相关
    ARK_API_KEY="YOUR_ARK_API_KEY"
    deepseek0324="YOUR_DEEPSEEK_MODEL_ENDPOINT_ID" # 例如: deepseek-chat

    # LangSmith (可选)
    ENABLE_LANGSMITH="true" # 或 "false"
    LangSmith_API_KEY="YOUR_LANGSMITH_API_KEY" # 如果 ENABLE_LANGSMITH 为 true
    LANGCHAIN_PROJECT="SQL_Agent_DeepSeek_Gradio" # LangSmith 项目名称 (可选)
    ```

    *   将 `YOUR_ARK_API_KEY` 替换为您的 VolcEngine Ark API 密钥。
    *   将 `YOUR_DEEPSEEK_MODEL_ENDPOINT_ID` 替换为您在 VolcEngine Ark 上使用的 DeepSeek 模型的 Endpoint ID。
    *   如果希望启用 LangSmith 追踪，请设置 `ENABLE_LANGSMITH="true"` 并提供 `LangSmith_API_KEY`。

## 运行应用

```bash
python sql_agent_app.py
```

应用启动后，将在本地启动一个 Gradio Web 服务器，通常在 `http://127.0.0.1:7860` 上可访问。

## 系统架构

该项目使用 LangGraph 构建了一个基于状态的 AI 工作流，主要包含以下组件：

1. **意图识别**：分析用户问题，确定查询意图
2. **数据库模式获取**：获取数据库结构信息
3. **SQL 生成**：根据用户问题和数据库结构生成 SQL 查询
4. **查询执行**：执行 SQL 查询并获取结果
5. **回答生成**：基于查询结果生成自然语言回答

## 状态定义

系统使用 `AgentState` 类型字典管理工作流状态：

```python
class AgentState(TypedDict):
    question: str  # 用户提出的问题
    thoughts: List[str]  # Agent 的思考过程
    intent: Optional[str]  # 用户意图
    sql_query: Optional[str]  # 生成的 SQL 查询
    sql_result: Optional[str]  # SQL 查询结果
    answer: Optional[str]  # 最终回答
    conversation_history: List[Dict[str, str]]  # 对话历史
    error: Optional[str]  # 错误信息
```

## 使用示例

用户可以通过 Gradio 界面输入自然语言问题，例如：

- "显示所有员工的姓名和职位"
- "哪些员工来自加拿大？"
- "谁是销售经理？"

系统会自动生成相应的 SQL 查询，执行查询并返回易于理解的回答。

## LangSmith 集成

如果启用了 LangSmith，系统会记录所有 LLM 调用和工作流执行情况，便于调试和优化。要启用 LangSmith，请确保在 `.env` 文件中设置了相关环境变量。

## 贡献指南

欢迎对本项目进行贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 许可证

[MIT License](LICENSE)

## 联系方式

如有任何问题或建议，请通过 [issue tracker](https://github.com/yourusername/your-repo/issues) 联系我们。