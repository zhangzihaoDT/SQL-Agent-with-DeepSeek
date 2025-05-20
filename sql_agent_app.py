import os
import gradio as gr
import sqlite3
from dotenv import load_dotenv
import time
import json
from typing import Dict, List, Tuple, Any, Optional, TypedDict, Union, Literal

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

# --- 配置 ---
ENV_PATH = "/Users/zihao_/Documents/coding/Langchain_chatwithdata/W20方向/.env"
DB_FILE = "chinook_agent.db"  # 将在脚本所在目录创建

# --- 加载环境变量并配置 LangSmith ---
load_dotenv(dotenv_path=ENV_PATH)

def configure_langsmith():
    if os.getenv("ENABLE_LANGSMITH", "false").lower() == "true":
        if os.getenv("LangSmith_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LangSmith_API_KEY")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "SQL_Agent_DeepSeek_Gradio")
            print(f"LangSmith 跟踪已启用，项目: {os.environ['LANGCHAIN_PROJECT']}")
            if not os.getenv("LANGCHAIN_API_KEY"):
                 print("警告: ENABLE_LANGSMITH 为 true，但 LangSmith_API_KEY 无法设置为 LANGCHAIN_API_KEY。跟踪可能无法工作。")
        else:
            print("警告: ENABLE_LANGSMITH 为 true，但在 .env 中找不到 LangSmith_API_KEY。跟踪已禁用。")
    else:
        print("LangSmith 跟踪已禁用。")

configure_langsmith()

# --- LLM 初始化 ---
def get_llm():
    ark_api_key = os.getenv("ARK_API_KEY")
    # 从 .env 文件获取 DeepSeek0324 的模型名称/端点 ID
    model_name_from_env = os.getenv("deepseek0324") 

    if not ark_api_key:
        raise ValueError("必须在 .env 文件中设置 ARK_API_KEY。")
    if not model_name_from_env:
        # 如果在 .env 中找不到 deepseek0324 的回退或错误
        print("警告: 在 .env 中找不到 'deepseek0324'，使用默认值 'Deepseek-V3'。这可能不是您想要的。")
        model_name_from_env = "Deepseek-V3"  # 通用回退，如果端点 ID 是必需的，则可能不起作用

    return ChatOpenAI(
        openai_api_key=ark_api_key,
        openai_api_base="https://ark.cn-beijing.volces.com/api/v3",  # 标准 VolcEngine Ark 基础 URL
        model_name=model_name_from_env,  # 使用 .env 中的值
        temperature=0
    )

# --- 数据库设置 ---
def setup_database(db_file=DB_FILE):
    db_path = os.path.join(os.path.dirname(__file__), db_file)  # 将数据库放在脚本旁边
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建一个简单的表（例如来自 Chinook 示例的 Employees 表）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Employees (
            EmployeeId INTEGER PRIMARY KEY,
            LastName TEXT,
            FirstName TEXT,
            Title TEXT,
            ReportsTo INTEGER,
            BirthDate TEXT,
            HireDate TEXT,
            Address TEXT,
            City TEXT,
            State TEXT,
            Country TEXT,
            PostalCode TEXT,
            Phone TEXT,
            Fax TEXT,
            Email TEXT
        )
    ''')
    # 如果表为空，则添加一些示例数据
    cursor.execute("SELECT COUNT(*) FROM Employees")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            (1, 'Adams', 'Andrew', 'General Manager', None, '1962-02-18', '2002-08-14', '11120 Jasper Ave NW', 'Edmonton', 'AB', 'Canada', 'T5K 2N1', '+1 (780) 428-9482', '+1 (780) 428-3457', 'andrew@chinookcorp.com'),
            (2, 'Edwards', 'Nancy', 'Sales Manager', 1, '1958-12-08', '2002-05-01', '825 8 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 2T3', '+1 (403) 262-3443', '+1 (403) 262-3322', 'nancy@chinookcorp.com'),
            (3, 'Peacock', 'Jane', 'Sales Support Agent', 2, '1973-08-29', '2002-04-01', '1111 6 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 5M5', '+1 (403) 262-3443', '+1 (403) 262-6712', 'jane@chinookcorp.com')
        ]
        cursor.executemany("INSERT INTO Employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", sample_data)
        print(f"已向 {db_file} 插入示例数据。")
    
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"

# --- 重构的 LangGraph Agent 实现 ---

# 定义 Agent 状态类型
class AgentState(TypedDict):
    question: str  # 用户提出的问题
    thoughts: List[str]  # Agent 的思考过程
    intent: Optional[str]  # 用户意图
    sql_query: Optional[str]  # 生成的 SQL 查询
    sql_result: Optional[str]  # SQL 查询结果
    answer: Optional[str]  # 最终回答
    conversation_history: List[Dict[str, str]]  # 对话历史
    error: Optional[str]  # 错误信息

# 提示模板
INTENT_RECOGNITION_PROMPT = """你是一个专业的数据库分析助手。请分析用户的问题，识别其意图。

用户问题: {question}

请确定用户是想要:
1. 获取表结构信息
2. 执行特定的 SQL 查询
3. 获取有关数据的一般信息

请只返回以下选项之一:
- GET_SCHEMA: 用户想了解数据库结构
- EXECUTE_QUERY: 用户想执行查询
- GET_INFO: 用户想获取一般信息
"""

SQL_GENERATION_PROMPT = """你是一个 SQL 专家。根据用户的问题生成适当的 SQL 查询。

数据库结构信息:
{schema}

用户问题: {question}

请生成一个能够回答用户问题的 SQL 查询。只返回 SQL 查询语句，不要有任何其他解释。
"""

ANSWER_GENERATION_PROMPT = """你是一个数据库分析助手。根据 SQL 查询结果回答用户的问题。

用户问题: {question}
SQL 查询: {sql_query}
查询结果: {sql_result}

请提供一个清晰、简洁的回答，解释查询结果并回答用户的问题。使用友好的语气，避免技术术语，除非必要。
"""

# 节点函数
def identify_intent(state: AgentState) -> AgentState:
    """识别用户意图"""
    llm = get_llm()
    
    intent_chain = ChatPromptTemplate.from_template(INTENT_RECOGNITION_PROMPT) | llm | StrOutputParser()
    
    intent = intent_chain.invoke({
        "question": state["question"]
    }).strip()
    
    thoughts = state.get("thoughts", [])
    thoughts.append(f"识别的意图: {intent}")
    
    return {
        **state,
        "thoughts": thoughts,
        "intent": intent
    }

def get_database_schema(state: AgentState) -> AgentState:
    """获取数据库模式"""
    db_uri = setup_database()
    db = SQLDatabase.from_uri(db_uri)
    
    schema = db.get_table_info()
    
    thoughts = state.get("thoughts", [])
    thoughts.append("获取了数据库模式")
    
    return {
        **state,
        "thoughts": thoughts,
        "schema": schema
    }

def generate_sql_query(state: AgentState) -> AgentState:
    """生成 SQL 查询"""
    llm = get_llm()
    
    sql_chain = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT) | llm | StrOutputParser()
    
    raw_sql_query = sql_chain.invoke({
        "schema": state.get("schema", ""),
        "question": state["question"]
    }).strip()
    
    # 清理 SQL 查询，移除可能的 Markdown 代码块标记
    sql_query = raw_sql_query
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1]
    if "```" in sql_query:
        sql_query = sql_query.split("```")[0]
    
    # 进一步清理和规范化 SQL 查询
    sql_query = sql_query.strip()
    
    thoughts = state.get("thoughts", [])
    thoughts.append(f"生成的 SQL 查询: {sql_query}")
    
    return {
        **state,
        "thoughts": thoughts,
        "sql_query": sql_query
    }

def execute_sql_query(state: AgentState) -> AgentState:
    """执行 SQL 查询"""
    db_uri = setup_database()
    db = SQLDatabase.from_uri(db_uri)
    
    try:
        # 确保 SQL 查询是干净的
        sql_query = state["sql_query"].strip()
        # 移除可能的 Markdown 代码块标记
        if "```" in sql_query:
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        sql_result = db.run(sql_query)
        
        thoughts = state.get("thoughts", [])
        thoughts.append("成功执行 SQL 查询")
        
        return {
            **state,
            "thoughts": thoughts,
            "sql_result": sql_result,
            "error": None
        }
    except Exception as e:
        thoughts = state.get("thoughts", [])
        thoughts.append(f"SQL 查询执行错误: {str(e)}")
        
        return {
            **state,
            "thoughts": thoughts,
            "error": f"SQL 查询执行错误: {str(e)}"
        }

def generate_answer(state: AgentState) -> AgentState:
    """生成最终回答"""
    llm = get_llm()
    
    # 如果有错误，直接返回错误信息
    if state.get("error"):
        return {
            **state,
            "answer": f"抱歉，我在执行查询时遇到了问题: {state['error']}"
        }
    
    answer_chain = ChatPromptTemplate.from_template(ANSWER_GENERATION_PROMPT) | llm | StrOutputParser()
    
    answer = answer_chain.invoke({
        "question": state["question"],
        "sql_query": state.get("sql_query", ""),
        "sql_result": state.get("sql_result", "")
    }).strip()
    
    thoughts = state.get("thoughts", [])
    thoughts.append("生成了最终回答")
    
    return {
        **state,
        "thoughts": thoughts,
        "answer": answer
    }

def direct_schema_response(state: AgentState) -> AgentState:
    """直接返回数据库模式信息"""
    db_uri = setup_database()
    db = SQLDatabase.from_uri(db_uri)
    
    schema = db.get_table_info()
    
    answer = f"以下是数据库的表结构信息:\n\n{schema}"
    
    thoughts = state.get("thoughts", [])
    thoughts.append("直接返回数据库模式信息")
    
    return {
        **state,
        "thoughts": thoughts,
        "answer": answer
    }

def route_by_intent(state: AgentState) -> str:
    """根据意图路由到不同的节点"""
    intent = state.get("intent", "")
    
    if intent == "GET_SCHEMA":
        return "direct_schema"
    elif intent == "EXECUTE_QUERY":
        return "query_flow"
    else:  # GET_INFO 或其他
        return "query_flow"  # 默认走查询流程

# 创建 SQL Agent
def create_sql_agent():
    """创建 SQL Agent 工作流"""
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("identify_intent", identify_intent)
    workflow.add_node("get_schema", get_database_schema)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_sql", execute_sql_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("direct_schema", direct_schema_response)
    
    # 设置入口点
    workflow.set_entry_point("identify_intent")
    
    # 添加边
    workflow.add_conditional_edges(
        "identify_intent",
        route_by_intent,
        {
            "direct_schema": "direct_schema",
            "query_flow": "get_schema"
        }
    )
    
    workflow.add_edge("get_schema", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_answer")
    
    # 设置终止节点
    workflow.add_edge("direct_schema", END)
    workflow.add_edge("generate_answer", END)
    
    # 编译工作流
    return workflow.compile()

# --- 初始化 Agent ---
try:
    agent_executor = create_sql_agent()
    print("SQL Agent 使用 LangGraph 初始化成功。")
except Exception as e:
    print(f"初始化期间出错: {e}")
    agent_executor = None

# --- Gradio 交互函数 ---
def query_agent(message: str, history: list):
    if agent_executor is None:
        yield "错误: SQL Agent 未能初始化。请检查您的 .env 配置和控制台日志。"
        return 
    
    if not message.strip(): 
        yield "请输入一个问题。"
        return

    try:
        # 准备对话历史
        conversation_history = []
        for human, ai in history:
            conversation_history.append({"role": "human", "content": human})
            conversation_history.append({"role": "ai", "content": ai})
        
        # 创建初始状态
        state = {
            "question": message,
            "thoughts": [],
            "intent": None,
            "sql_query": None,
            "sql_result": None,
            "answer": None,
            "conversation_history": conversation_history,
            "error": None
        }
        
        # 调用 Agent
        response = agent_executor.invoke(state)
        
        # 提取最终答案
        final_answer = response.get("answer", "未能生成回答。")
        
        # 提取中间步骤
        thoughts = response.get("thoughts", [])
        sql_query = response.get("sql_query", "")
        sql_result = response.get("sql_result", "")
        
        # 构建详细信息
        details = []
        if thoughts:
            details.append("**思考过程:**\n" + "\n".join([f"- {thought}" for thought in thoughts]))
        
        if sql_query:
            details.append(f"**SQL 查询:**\n```sql\n{sql_query}\n```")
        
        if sql_result:
            details.append(f"**查询结果:**\n```\n{sql_result}\n```")
        
        # 构造完整的响应字符串
        full_response_str = final_answer
        
        if details:
            details_content = "\n\n".join(details)
            details_html = f"\n\n<details>\n<summary><strong>--- Agent 步骤和 SQL 查询 (点击展开) ---</strong></summary>\n\n{details_content}\n</details>\n"
            full_response_str += details_html
        
        # 逐字流式传输完整的响应字符串
        buffer = ""
        for char_token in full_response_str:
            buffer += char_token
            time.sleep(0.01)  # 调整打字速度
            yield buffer

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Agent 调用期间出错: {error_details}") 
        error_message = (f"发生错误: {str(e)}\n\n"
                         "详细信息:\n"
                         f"{error_details}\n\n"
                         "故障排除提示:\n"
                         "- 确保您的 .env 文件具有正确的 ARK_API_KEY 和 deepseek0324 模型 ID。\n"
                         "- 验证 DeepSeek 模型版本是否支持工具使用（函数调用）。\n"
                         "- 检查控制台日志以获取更具体的来自 Langchain 或 LLM API 的错误消息。")
        yield error_message

# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SQL 查询助手 (基于 LangGraph 和 DeepSeek)
        使用 LangGraph 和 DeepSeek 模型查询 SQLite 数据库 (Employees 表示例)。
        """
    )
    gr.Markdown(
        f"""
        LLM 配置从 '{os.path.basename(ENV_PATH)}' 加载。
        数据库文件: '{DB_FILE}'。
        Agent 会显示最终答案以及可折叠的 SQL 查询和中间步骤。
        """
    )
    
    chat_interface = gr.ChatInterface(
        fn=query_agent,
        examples=[ 
            "描述 Employees 表",
            "加拿大有多少员工？",
            "卡尔加里的员工有哪些？",
            "列出所有员工",
            "Employees 表中有哪些不同的国家？"
        ],
        chatbot=gr.Chatbot(height=500, show_copy_button=True), 
        textbox=gr.Textbox(placeholder="请输入您关于数据库的问题...", container=False, scale=7),
        retry_btn="🔄 重试",
        undo_btn="↩️ 撤销",
        clear_btn="🗑️ 清除对话",
    )

if __name__ == "__main__":
    print(f"尝试启动 Gradio 界面...")
    demo.launch()