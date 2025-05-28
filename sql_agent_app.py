# 在文件开头的导入部分添加
import os
import gradio as gr
import sqlite3
import duckdb  # 添加 DuckDB 导入
import sqlalchemy  # 添加 SQLAlchemy 导入
from sqlalchemy import inspect, MetaData, Table  # 导入可能需要的特定功能
# 关闭警告 ：如果索引反射警告不重要，可以禁用它：
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="duckdb_engine")

from dotenv import load_dotenv
import time
import json
import re  # 确保导入re模块用于正则表达式
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
# 更新数据库文件路径
DB_FILE = "/Users/zihao_/Documents/coding/Langchain_chatwithdata/database/central_analytics.duckdb"

# --- 初始化 Agent ---
# 全局变量存储表结构信息
_TABLE_STRUCTURE = {}
# 全局数据库连接
_DB_CONNECTION = None

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
def get_llm(model_type="deepseek"):
    """获取LLM模型，支持多模型架构
    
    Args:
        model_type: 模型类型，可选值："deepseek"或"glm4"
    
    Returns:
        ChatOpenAI: 配置好的LLM模型实例
    """
    if model_type == "deepseek":
        # DeepSeek模型配置
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
    elif model_type == "glm4":
        # GLM4-Flash模型配置
        glm4_api_key = os.getenv("glm4_AI_KEY")
        
        if not glm4_api_key:
            raise ValueError("必须在 .env 文件中设置 glm4_AI_KEY。")
            
        return ChatOpenAI(
            temperature=0,
            model="GLM-4-Flash-250414",
            openai_api_key=glm4_api_key,
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# --- 数据库设置 ---
def setup_database(db_file=DB_FILE, verbose=False):
    # 使用正确的 SQLAlchemy URI 格式
    # 对于 DuckDB，格式应该是 "duckdb:///:memory:" 或 "duckdb:///path/to/file"
    if verbose:
        if os.path.exists(db_file):
            print(f"连接到现有数据库: {db_file}")
        else:
            print(f"警告: 数据库文件 '{db_file}' 不存在!")
    
    # 注意这里使用三个斜杠，这是 SQLAlchemy 的要求
    return f"duckdb:///{db_file}"

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

# 共享提示片段
SQL_BASE_GUIDELINES = """
请注意以下几点：
1. 必须使用上面提供的数据库结构中实际存在的表名和列名
2. 表名和列名可能包含中文或特殊字符，请使用双引号将它们括起来
3. 不要使用 Markdown 格式（如 ```sql）包装你的查询
4. 确保查询语法与 DuckDB 兼容
5. 如果查询涉及多个表，请确保表之间的关系正确
"""

# 更新SQL_GENERATION_PROMPT
SQL_GENERATION_PROMPT = f"""你是一个 SQL 专家。根据用户的问题生成适当的 SQL 查询。

数据库是 DuckDB，它与 PostgreSQL 语法兼容，但有一些特殊功能。

数据库结构信息:
{{schema}}

用户问题: {{question}}

{SQL_BASE_GUIDELINES}
1. 车型名必须带品牌前缀，如 `"蔚来ET7"` 而非 `"ET7"`
2. 查询车型时必须使用模糊匹配：`"车型" LIKE '%蔚来ES6%'`
3. 当需要精确匹配时，可以结合品牌条件，如 `"品牌" = '蔚来' AND "车型" LIKE '%ES6%'`
4. 对于可能有多个版本的车型，模糊匹配可以获取所有相关版本
5. 销量数据可能存储在"量"列中，而不是"销量"列

只返回 SQL 查询语句，不要有任何其他解释或格式标记。
"""

ANSWER_GENERATION_PROMPT = """你是一个数据库分析助手。根据 SQL 查询结果回答用户的问题。

用户问题: {question}
SQL 查询: {sql_query}
查询结果: {sql_result}

请提供一个清晰、简洁的回答，解释查询结果并回答用户的问题。使用友好的语气，避免技术术语，除非必要。
"""

# 节点函数
def identify_intent(state: AgentState) -> AgentState:
    """识别用户意图，使用GLM4-Flash模型"""
    # 使用GLM4模型
    llm = get_llm("glm4")
    
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

def get_db_connection(verbose=False):
    """获取数据库连接（单例模式）"""
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        db_uri = setup_database(verbose=verbose)
        _DB_CONNECTION = SQLDatabase.from_uri(db_uri)
        if verbose:
            print("创建了新的数据库连接")
    return _DB_CONNECTION

def get_database_schema(state: AgentState) -> AgentState:
    """获取数据库模式"""
    db = get_db_connection()
    schema = db.get_table_info()
    
    thoughts = state.get("thoughts", [])
    thoughts.append("获取了数据库模式")
    
    return {
        **state,
        "thoughts": thoughts,
        "schema": schema
    }

def generate_sql_query(state: AgentState) -> AgentState:
    """生成 SQL 查询，使用DeepSeek模型"""
    # 明确指定使用DeepSeek模型
    llm = get_llm("deepseek")
    
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

def validate_sql_query(state: AgentState) -> AgentState:
    """验证 SQL 查询，使用DeepSeek模型"""
    # 明确指定使用DeepSeek模型
    llm = get_llm("deepseek")
    
    # 获取数据库中实际存在的表名列表和列名信息
    db = get_db_connection()
    actual_tables = db.get_usable_table_names()
    
    # 获取每个表的列信息
    table_columns = {}
    for table in actual_tables:
        try:
            # 使用全局表结构信息
            global _TABLE_STRUCTURE
            if table in _TABLE_STRUCTURE:
                table_columns[table] = [col["name"] for col in _TABLE_STRUCTURE[table]]
            else:
                # 如果全局表结构中没有，尝试直接查询
                columns_query = f'DESCRIBE "{table}"'
                columns_result = db.run(columns_query)
                # 提取列名
                column_names = []
                for line in columns_result.strip().split('\n'):
                    if line and '|' in line:
                        # 第一列通常是列名
                        column_name = line.split('|')[0].strip()
                        if column_name and column_name != "column_name" and not column_name.startswith('-'):
                            column_names.append(column_name)
                table_columns[table] = column_names
        except Exception as e:
            print(f"获取表 {table} 的列信息时出错: {e}")
    
    validation_prompt = """你是一个 SQL 专家。请验证以下 SQL 查询是否有效，并修复任何问题。

    数据库是 DuckDB，它与 PostgreSQL 语法兼容。

    数据库结构信息:
    {schema}

    数据库中实际存在的表:
    {actual_tables}

    每个表的列信息:
    {table_columns}

    原始 SQL 查询:
    {sql_query}

    请检查以下问题：
    1. 表名必须是数据库中实际存在的表，不要使用不存在的表名
    2. 列名必须是表中实际存在的列，不要使用不存在的列名
    3. 表名和列名是否正确引用（特别是包含中文或特殊字符的名称）
    4. SQL 语法是否正确
    5. 查询是否与数据库结构匹配
    6. 日期类型的列处理是否正确，特别是"日期 年"和"日期 月"等特殊命名的列

    只返回修复后的 SQL 查询，不要有任何其他解释或格式标记。如果原始查询已经正确，则直接返回原始查询。
    """
    
    # 格式化表列信息为易读的文本
    table_columns_text = ""
    for table, columns in table_columns.items():
        quoted_columns = [f'"{col}"' for col in columns]
        table_columns_text += f'表 "{table}" 的列: {", ".join(quoted_columns)}\n'
    
    validation_chain = ChatPromptTemplate.from_template(validation_prompt) | llm | StrOutputParser()
    
    validated_sql = validation_chain.invoke({
        "schema": state.get("schema", ""),
        "actual_tables": ", ".join([f'"{table}"' for table in actual_tables]),
        "table_columns": table_columns_text,
        "sql_query": state.get("sql_query", "")
    }).strip()
    
    # 清理验证后的 SQL
    if "```" in validated_sql:
        validated_sql = validated_sql.replace("```sql", "").replace("```", "").strip()
    
    # 确保返回的是有效的 SQL 查询，而不是解释文本
    if validated_sql.lower().startswith("select") or validated_sql.lower().startswith("with") or validated_sql.lower().startswith("update") or validated_sql.lower().startswith("delete") or validated_sql.lower().startswith("insert"):
        # 这是一个有效的 SQL 查询
        thoughts = state.get("thoughts", [])
        if validated_sql != state.get("sql_query", ""):
            thoughts.append(f"SQL 查询已修正: {validated_sql}")
        else:
            thoughts.append("SQL 查询验证通过，无需修改")
        
        return {
            **state,
            "thoughts": thoughts,
            "sql_query": validated_sql
        }
    else:
        # 返回原始查询，因为验证结果不是有效的 SQL
        thoughts = state.get("thoughts", [])
        thoughts.append("验证结果不是有效的 SQL 查询，使用原始查询")
        
        return {
            **state,
            "thoughts": thoughts,
            "sql_query": state.get("sql_query", "")
        }

# 创建 SQL Agent
def create_sql_agent():
    """创建 SQL Agent 工作流"""
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("identify_intent", identify_intent)
    workflow.add_node("get_schema", get_database_schema)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("validate_sql", validate_sql_query)  # 新增验证节点
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
    workflow.add_edge("generate_sql", "validate_sql")  # 添加到验证节点的边
    workflow.add_edge("validate_sql", "execute_sql")   # 从验证节点到执行节点的边
    workflow.add_edge("execute_sql", "generate_answer")
    
    # 设置终止节点
    workflow.add_edge("direct_schema", END)
    workflow.add_edge("generate_answer", END)
    
    # 编译工作流
    return workflow.compile()

def execute_sql_query(state: AgentState) -> AgentState:
    """执行 SQL 查询"""
    # 使用连接池而不是创建新连接
    db = get_db_connection()
    
    try:
        # 更彻底地清理 SQL 查询
        sql_query = state["sql_query"].strip()
        
        # 移除所有可能的 Markdown 代码块标记和其他非 SQL 内容
        if "```" in sql_query:
            # 提取 ``` 之间的内容
            import re
            code_blocks = re.findall(r'```(?:sql)?(.*?)```', sql_query, re.DOTALL)
            if code_blocks:
                sql_query = code_blocks[0].strip()
            else:
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # 确保表名和列名中的中文或特殊字符被正确引用
        thoughts = state.get("thoughts", [])
        thoughts.append(f"清理后的 SQL 查询: {sql_query}")
        
        # 执行查询
        sql_result = db.run(sql_query)
        
        thoughts.append("成功执行 SQL 查询")
        
        return {
            **state,
            "thoughts": thoughts,
            "sql_result": sql_result,
            "error": None
        }
    except Exception as e:
        thoughts = state.get("thoughts", [])
        error_msg = str(e)
        thoughts.append(f"SQL 查询执行错误: {error_msg}")
        
        # 提供更详细的错误信息和可能的解决方案
        error_analysis = "未知错误"
        suggested_fix = ""
        
        if "syntax error" in error_msg.lower():
            error_analysis = "SQL 语法错误，请检查查询语法"
        elif "no such table" in error_msg.lower():
            # 提取错误中提到的表名
            match = re.search(r'Table with name (.*?) does not exist', error_msg)
            if match:
                wrong_table = match.group(1)
                # 获取可能的替代表
                actual_tables = db.get_usable_table_names()
                suggested_tables = []
                for table in actual_tables:
                    if wrong_table.lower() in table.lower():
                        suggested_tables.append(table)
                
                if suggested_tables:
                    suggested_fix = f"可能的替代表: {', '.join(suggested_tables)}"
                
            error_analysis = "表不存在，请检查表名是否正确（注意表名可能包含中文或特殊字符）"
        elif "no such column" in error_msg.lower() or "column" in error_msg.lower() and "does not exist" in error_msg.lower() or "not found in FROM clause" in error_msg.lower():
            # 提取错误中提到的列名和表名
            col_match = re.search(r'column ["\']?(.*?)["\']? does not exist|Referenced column ["\']?(.*?)["\']? not found', error_msg)
            table_match = re.search(r'FROM ["\']?(.*?)["\']?', sql_query)
            
            if col_match:
                wrong_col = col_match.group(1) or col_match.group(2)
                if table_match:
                    table_name = table_match.group(1)
                    # 获取表的列信息
                    try:
                        columns_query = f'DESCRIBE "{table_name}"'
                        columns_result = db.run(columns_query)
                        # 提取列名
                        column_names = []
                        for line in columns_result.strip().split('\n'):
                            if line and '|' in line:
                                column_name = line.split('|')[0].strip()
                                if column_name and column_name != "column_name" and not column_name.startswith('-'):
                                    column_names.append(column_name)
                        
                        # 找出相似的列名
                        similar_cols = []
                        for col in column_names:
                            if wrong_col.lower() in col.lower() or col.lower() in wrong_col.lower():
                                similar_cols.append(col)
                        
                        if similar_cols:
                            suggested_fix = f"表 '{table_name}' 中可能的替代列: {', '.join(similar_cols)}"
                        else:
                            suggested_fix = f"表 '{table_name}' 的所有列: {', '.join(column_names)}"
                    except Exception as col_err:
                        print(f"获取列信息时出错: {col_err}")
            
            error_analysis = "列不存在，请检查列名是否正确（注意列名可能包含中文或特殊字符）"
        
        error_message = f"SQL 查询执行错误: {error_msg}\n可能的原因: {error_analysis}"
        if suggested_fix:
            error_message += f"\n{suggested_fix}"
        
        return {
            **state,
            "thoughts": thoughts,
            "error": error_message
        }

def generate_answer(state: AgentState) -> AgentState:
    """生成最终回答，使用GLM4-Flash模型"""
    # 使用GLM4模型
    llm = get_llm("glm4")
    
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
    """直接返回数据库模式信息，使用GLM4-Flash模型，并能回答关于表结构的问题"""
    # 使用GLM4模型
    llm = get_llm("glm4")
    
    # 使用连接池而不是创建新连接
    db = get_db_connection()
    
    # 获取数据库模式信息
    schema = db.get_table_info()
    
    # 获取全局表结构信息
    global _TABLE_STRUCTURE
    if not _TABLE_STRUCTURE:
        _TABLE_STRUCTURE = analyze_database_structure()
    
    # 格式化表结构信息为易读文本
    table_structure_text = format_table_structure(_TABLE_STRUCTURE)
    
    # 检查用户问题是否针对特定表
    user_question = state["question"]
    specific_table_info = ""
    
    # 检查用户是否询问特定表的信息
    for table_name in _TABLE_STRUCTURE.keys():
        if table_name.lower() in user_question.lower():
            specific_table_info = f"关于表 \"{table_name}\" 的详细信息:\n"
            specific_table_info += f"列数: {len(_TABLE_STRUCTURE[table_name])}\n"
            specific_table_info += "列名和类型:\n"
            for col in _TABLE_STRUCTURE[table_name]:
                specific_table_info += f"  - \"{col['name']}\" ({col['type']})\n"
            break
    
    # 使用GLM4模型生成更友好的模式描述
    schema_prompt = f"""你是一个数据库专家。请根据以下数据库模式信息和表结构详情，回答用户的问题。

    用户问题: {user_question}
    
    数据库模式信息:
    {schema}
    
    表结构详细信息:
    {table_structure_text}
    
    {specific_table_info if specific_table_info else ""}
    
    请提供一个友好的回答，避免技术术语，除非必要。如果用户询问特定表的信息，请重点介绍该表的结构和用途。
    如果用户询问的表不存在，请告知用户并列出可用的表。"""
    
    schema_chain = ChatPromptTemplate.from_template(schema_prompt) | llm | StrOutputParser()
    
    answer = schema_chain.invoke({}).strip()
    
    thoughts = state.get("thoughts", [])
    thoughts.append("使用GLM4生成了数据库模式描述并回答了关于表结构的问题")
    
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

def analyze_database_structure(verbose=False):
    """分析数据库结构，提取表和列信息"""
    global _TABLE_STRUCTURE
    
    if _TABLE_STRUCTURE:
        return _TABLE_STRUCTURE
    
    db = get_db_connection(verbose)
    tables = db.get_usable_table_names()
    _TABLE_STRUCTURE = {}
    
    # 直接使用 SQLAlchemy 的反射功能获取表结构
    engine = db._engine
    inspector = sqlalchemy.inspect(engine)
    
    for table in tables:
        try:
            columns = inspector.get_columns(table)
            _TABLE_STRUCTURE[table] = [{"name": col["name"], "type": str(col["type"])} for col in columns]
            
            if verbose:
                print(f"分析表 '{table}': 找到 {len(_TABLE_STRUCTURE[table])} 列")
                if _TABLE_STRUCTURE[table]:
                    print(f"列名示例: {', '.join([col['name'] for col in _TABLE_STRUCTURE[table][:3]])}")
        except Exception as e:
            if verbose:
                print(f"分析表 '{table}' 结构时出错: {str(e)}")
    
    return _TABLE_STRUCTURE

# 在 Agent 初始化时调用
try:
    # 分析数据库结构
    table_structure = analyze_database_structure(verbose=True)
    print(f"成功分析了 {len(table_structure)} 个表的结构")
    
    agent_executor = create_sql_agent()
    print("SQL Agent 使用 LangGraph 初始化成功。")
except Exception as e:
    print(f"初始化期间出错: {e}")
    agent_executor = None
    table_structure = {}

# --- Gradio 交互函数 ---
def format_table_structure(table_structure: dict) -> str:
    """格式化表结构信息为易读文本"""
    table_structure_text = ""
    for table, columns in table_structure.items():
        table_structure_text += f'表 "{table}" 的列:\n'
        for col in columns:
            table_structure_text += f'  - "{col["name"]}" ({col["type"]})\n'
        table_structure_text += "\n"
    return table_structure_text

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
        使用 LangGraph 和 DeepSeek 模型查询 DuckDB 数据库。
        """
    )
    gr.Markdown(
        f"""
        LLM 配置从 '{os.path.basename(ENV_PATH)}' 加载。
        数据库文件: '{os.path.basename(DB_FILE)}'。
        Agent 会显示最终答案以及可折叠的 SQL 查询和中间步骤。
        """
    )
    
    chat_interface = gr.ChatInterface(
        fn=query_agent,
        examples=[ 
            "数据库中有哪些表？",
            '显示"上险数_03_data_截止 202504_clean"表中的前5条记录',
            "智己2024年销量如何？",
            "智己LS62024年月度销量走势？",
            "新能源市场月度渗透率走势",
            "2024年不同燃料类型车型数量是多少？",
            "2024年不同燃料类型车型数量是多少？从价格配置表中查询。",
            "在价格配置表中找出2024年20-25万增程（REEV）suv销量前十车型",
            "2024年哪个城市级别的销量最高？",
            "对比蔚来ET5 和智己L6用户年龄百分比分布",
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