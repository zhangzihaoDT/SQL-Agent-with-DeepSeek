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
    question: str
    thoughts: List[str]
    intent: Optional[str]
    schema: Optional[str]
    complexity: Optional[str]  # 新增：复杂度评估
    strategy: Optional[str]    # 新增：处理策略
    planning_output: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[str]
    answer: Optional[str]
    conversation_history: List[Dict[str, str]]
    error: Optional[str]

# 提示模板
# 智能路由：合并的提示模板，一次性完成多个判断
intelligent_routing_prompt = """你是一个专业的数据库分析助手和查询规划师。请分析用户的问题，完成以下三个任务：

用户问题: {question}
数据库Schema: {schema}

请按以下格式输出你的分析结果：

**1. 意图识别:**
请确定用户是想要:
- GET_SCHEMA: 用户想了解数据库结构
- EXECUTE_QUERY: 用户想执行查询
- GET_INFO: 用户想获取一般信息

**2. 复杂度评估:**
分析问题复杂度（简单/中等/复杂）：
- 简单：单表查询，基本条件筛选
- 中等：涉及聚合、排序、简单计算
- 复杂：多表关联、复杂分析、对比研究

**3. 处理策略:**
基于以上分析，建议的处理路径：
- DIRECT_SCHEMA: 直接返回表结构信息
- SIMPLE_QUERY: 直接生成SQL，无需详细规划
- COMPLEX_QUERY: 需要详细规划后再生成SQL

**4. 规划输出:**
如果是复杂查询，请提供简要的查询规划；如果是简单查询，说明"无需详细规划"。

请严格按照上述格式输出，每个部分用一行简洁的文字说明。
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
SQL_GENERATION_PROMPT = f"""你是一个 SQL 专家。根据用户的问题和提供的查询计划，生成适当的 SQL 查询。

数据库是 DuckDB，它与 PostgreSQL 语法兼容，但有一些特殊功能。

你可以一步一步来分析需求，最终确定需要生成的查询SQL。

数据库结构信息:
{{schema}}

用户问题: {{question}}

查询规划:
{{planning_output}}

{SQL_BASE_GUIDELINES}

特定场景处理指南:
1.  **车型名称处理**:
    *   车型名必须带品牌前缀，例如 "蔚来ET7" 而非 "ET7"。
    *   查询车型时必须使用模糊匹配：`"车型" LIKE '%蔚来ES6%'`。
    *   当需要精确匹配时，可以结合品牌条件，例如 `"品牌" = '蔚来' AND "车型" LIKE '%ES6%'`。
    *   对于可能有多个版本的车型，模糊匹配可以获取所有相关版本。

2.  **百分比分布和聚合查询**:
    *   如果用户问题涉及到计算不同类别（如年龄段、性别等）的百分比分布，特别是在对比多个实体（如不同车型）时，你需要生成能够直接在数据库层面完成聚合和百分比计算的SQL。
    *   例如，如果问题是“对比A车型和B车型用户在不同年龄段的百分比分布”，并且数据库中年龄是具体数值，你需要：
        a.  根据业务逻辑定义年龄段（例如：20-30岁, 31-40岁等），可以使用 `CASE WHEN ... THEN ... ELSE ... END` 语句创建临时的年龄段列。
        b.  按车型和年龄段对用户数量进行分组计数 (`COUNT(*)` 或 `COUNT(DISTINCT user_id)`).
        c.  计算每个年龄段在对应车型用户总数中的百分比。这通常涉及到子查询或窗口函数来获取每个车型的总用户数，然后计算 `(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY "车型"))`。
    *   确保查询结果直接包含车型、年龄段以及对应的百分比，例如：`车型 | 年龄段 | 用户百分比`。

3.  **多条SQL查询处理**:
    *   如果查询规划中建议了多条SQL查询，并且规划明确指出需要多条，请只生成第一条建议的SQL查询。后续的查询将在其他步骤处理。

只返回 SQL 查询语句，不要有任何其他解释或格式标记。
"""

ANSWER_GENERATION_PROMPT = """你是一个高级数据分析师。根据 SQL 查询结果回答用户的问题。

用户问题: {question}
SQL 查询: {sql_query}
查询结果: {sql_result}

**重要指令：**
1. **严格基于查询结果**：你必须严格基于上面提供的 `查询结果` 来回答用户问题。绝对不要编造、推测或使用你的预训练知识来填充数据。

2. **处理空结果**：
   - 如果 `查询结果` 为空、显示"Empty DataFrame"、"no rows"或类似内容，请明确告知用户："根据您的查询条件，在数据库中未找到符合条件的数据。"
   - 不要提供任何虚构的数据或示例。

3. **数据准确性**：
   - 如果查询结果包含具体数据，请准确引用其中的数值、名称和其他信息。
   - 对于数值数据，请使用查询结果中的确切数字，不要四舍五入或估算。
   - 对于列表或排名，请按查询结果中的确切顺序和内容呈现。

4. **格式化输出**：
   - 将查询结果转换为清晰、易读的自然语言描述。
   - 使用友好的语气，但保持专业性。
   - 避免过度的技术术语，除非用户明确需要。

5. **结果验证**：
   - 在回答之前，请仔细检查你的回答是否与 `查询结果` 中的信息完全一致。
   - 如果查询结果的格式难以理解，请尽力解释，但不要添加不存在的信息。

请基于以上指令，提供一个准确、清晰的回答。
"""

# 节点函数
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

def intelligent_router(state: AgentState) -> AgentState:
    """智能路由节点：同时完成意图识别、复杂度判断和规划决策"""
    llm = get_llm("glm4")
    
    routing_chain = ChatPromptTemplate.from_template(intelligent_routing_prompt) | llm | StrOutputParser()
    
    routing_result = routing_chain.invoke({
        "question": state["question"],
        "schema": state.get("schema", "无法获取Schema信息")
    }).strip()
    
    # 解析路由结果
    intent = "EXECUTE_QUERY"  # 默认值
    complexity = "simple"     # 默认值
    strategy = "SIMPLE_QUERY" # 默认值
    planning_output = "无需详细规划"
    
    try:
        lines = routing_result.split('\n')
        for line in lines:
            if "意图识别" in line or "GET_SCHEMA" in line:
                if "GET_SCHEMA" in line:
                    intent = "GET_SCHEMA"
                elif "EXECUTE_QUERY" in line:
                    intent = "EXECUTE_QUERY"
                elif "GET_INFO" in line:
                    intent = "GET_INFO"
            
            elif "处理策略" in line or "DIRECT_SCHEMA" in line or "SIMPLE_QUERY" in line or "COMPLEX_QUERY" in line:
                if "DIRECT_SCHEMA" in line:
                    strategy = "DIRECT_SCHEMA"
                elif "COMPLEX_QUERY" in line:
                    strategy = "COMPLEX_QUERY"
                elif "SIMPLE_QUERY" in line:
                    strategy = "SIMPLE_QUERY"
            
            elif "复杂度评估" in line:
                if "复杂" in line:
                    complexity = "complex"
                elif "中等" in line:
                    complexity = "medium"
                else:
                    complexity = "simple"
        
        # 提取规划输出
        if "规划输出" in routing_result:
            planning_section = routing_result.split("规划输出:")[1].strip()
            if planning_section and "无需详细规划" not in planning_section:
                planning_output = planning_section
    
    except Exception as e:
        print(f"解析路由结果时出错: {e}")
        # 使用默认值
    
    thoughts = state.get("thoughts", [])
    thoughts.append(f"智能路由分析 - 意图: {intent}, 复杂度: {complexity}, 策略: {strategy}")
    thoughts.append(f"路由详细结果: {routing_result}")
    
    return {
        **state,
        "thoughts": thoughts,
        "intent": intent,
        "complexity": complexity,
        "strategy": strategy,
        "planning_output": planning_output
    }

def route_by_strategy(state: AgentState) -> str:
    """根据智能路由的策略决定下一步"""
    strategy = state.get("strategy", "SIMPLE_QUERY")
    
    if strategy == "DIRECT_SCHEMA":
        return "direct_schema"
    elif strategy == "COMPLEX_QUERY":
        return "generate_sql"  # 复杂查询直接生成SQL，因为已经有了规划
    else:  # SIMPLE_QUERY
        return "generate_sql"  # 简单查询直接生成SQL
    
def generate_sql_query(state: AgentState) -> AgentState:
    """生成 SQL 查询，使用DeepSeek模型"""
    # 明确指定使用DeepSeek模型
    llm = get_llm("deepseek")
    
    sql_chain = ChatPromptTemplate.from_template(SQL_GENERATION_PROMPT) | llm | StrOutputParser()
    
    raw_sql_query = sql_chain.invoke({
        "schema": state.get("schema", "无法获取Schema信息"),
        "question": state["question"],
        "planning_output": state.get("planning_output", "无规划信息") # 传递规划输出
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
    """验证和修复 SQL 查询 - 使用缓存的表结构"""
    llm = get_llm("deepseek")
    db = get_db_connection()
    
    # 确保表结构已缓存
    global _TABLE_STRUCTURE
    if not _TABLE_STRUCTURE:
        _TABLE_STRUCTURE = analyze_database_structure(verbose=True)
    
    # 获取实际存在的表名
    actual_tables = list(_TABLE_STRUCTURE.keys())
    
    # 直接从缓存中获取列信息，无需查询数据库
    table_columns = {}
    for table in actual_tables:
        if table in _TABLE_STRUCTURE:
            table_columns[table] = [col["name"] for col in _TABLE_STRUCTURE[table]]
        else:
            # 这种情况理论上不应该发生，因为我们已经完整缓存了
            print(f"警告: 表 '{table}' 不在缓存中，这可能表示缓存不完整")
            table_columns[table] = []
    
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

# 添加缓存状态监控函数
def get_cache_status():
    """获取表结构缓存状态"""
    global _TABLE_STRUCTURE
    if not _TABLE_STRUCTURE:
        return {
            "cached": False,
            "table_count": 0,
            "total_columns": 0
        }
    
    total_columns = sum(len(columns) for columns in _TABLE_STRUCTURE.values())
    return {
        "cached": True,
        "table_count": len(_TABLE_STRUCTURE),
        "total_columns": total_columns,
        "tables": list(_TABLE_STRUCTURE.keys())
    }

def refresh_table_structure_cache(verbose=False):
    """强制刷新表结构缓存"""
    global _TABLE_STRUCTURE
    _TABLE_STRUCTURE = {}
    return analyze_database_structure(verbose=verbose)


# 修改 create_sql_agent 函数
def create_sql_agent():
    """创建优化后的 SQL Agent 工作流"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("get_schema", get_database_schema)
    workflow.add_node("intelligent_router", intelligent_router)  # 新的智能路由节点
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("validate_sql", validate_sql_query)
    workflow.add_node("execute_sql", execute_sql_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("direct_schema", direct_schema_response)
    
    # 设置入口点
    workflow.set_entry_point("get_schema")
    
    # 简化的边连接
    workflow.add_edge("get_schema", "intelligent_router")
    
    # 根据智能路由的策略进行条件路由
    workflow.add_conditional_edges(
        "intelligent_router",
        route_by_strategy,
        {
            "direct_schema": "direct_schema",
            "generate_sql": "generate_sql"
        }
    )
    
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_edge("validate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_answer")
    
    # 设置结束点
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("direct_schema", END)
    
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
        
        # 检查查询结果是否为空，并提供明确的标识
        if not sql_result or sql_result.strip() == "" or "Empty DataFrame" in str(sql_result) or sql_result.strip() == "[]":
            sql_result = "[查询执行成功，但未找到符合条件的数据记录]"
            thoughts.append("SQL 查询执行成功，但返回空结果")
        else:
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
    
    # 如果有错误，让LLM解释错误信息
    if state.get("error"):
        error_message = state["error"]
        error_explanation_prompt = f"""你是一个乐于助人的AI助手。用户在尝试执行SQL查询时遇到了以下错误：

        错误信息：
        {error_message}

        请简短，解释这个错误，并尽可能提供一些建议来帮助用户解决问题。
"""
        
        explanation_chain = ChatPromptTemplate.from_template(error_explanation_prompt) | llm | StrOutputParser()
        
        explained_error = explanation_chain.invoke({}).strip()
        
        thoughts = state.get("thoughts", [])
        thoughts.append(f"LLM解释了SQL执行错误: {error_message}")
        
        return {
            **state,
            "thoughts": thoughts,
            "answer": explained_error
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

def analyze_database_structure(verbose=False):
    """分析数据库结构，提取表和列信息 - 启动时完整缓存版本"""
    global _TABLE_STRUCTURE
    
    if _TABLE_STRUCTURE:
        if verbose:
            print(f"使用已缓存的表结构信息，包含 {len(_TABLE_STRUCTURE)} 个表")
        return _TABLE_STRUCTURE
    
    if verbose:
        print("开始分析数据库结构并建立完整缓存...")
    
    db = get_db_connection(verbose)
    tables = db.get_usable_table_names()
    _TABLE_STRUCTURE = {}
    
    # 使用 SQLAlchemy 的反射功能批量获取所有表结构
    engine = db._engine
    inspector = sqlalchemy.inspect(engine)
    
    successful_tables = 0
    failed_tables = []
    
    for table in tables:
        try:
            columns = inspector.get_columns(table)
            _TABLE_STRUCTURE[table] = [
                {"name": col["name"], "type": str(col["type"])} 
                for col in columns
            ]
            successful_tables += 1
            
            if verbose:
                print(f"✓ 表 '{table}': {len(_TABLE_STRUCTURE[table])} 列")
                
        except Exception as e:
            failed_tables.append((table, str(e)))
            if verbose:
                print(f"✗ 表 '{table}' 分析失败: {str(e)}")
    
    if verbose:
        print(f"\n数据库结构分析完成:")
        print(f"  - 成功分析: {successful_tables} 个表")
        print(f"  - 失败: {len(failed_tables)} 个表")
        if failed_tables:
            print(f"  - 失败的表: {[table for table, _ in failed_tables]}")
    
    return _TABLE_STRUCTURE

# 在 Agent 初始化时调用 - 更新版本
try:
    print("正在初始化 SQL Agent...")
    
    # 预先分析并缓存所有数据库结构
    print("步骤 1: 分析数据库结构并建立缓存...")
    table_structure = analyze_database_structure(verbose=True)
    
    if table_structure:
        cache_status = get_cache_status()
        print(f"✓ 表结构缓存建立成功:")
        print(f"  - 缓存表数量: {cache_status['table_count']}")
        print(f"  - 总列数: {cache_status['total_columns']}")
        print(f"  - 表列表: {', '.join(cache_status['tables'][:5])}{'...' if len(cache_status['tables']) > 5 else ''}")
    else:
        print("⚠️  警告: 未能缓存任何表结构")
    
    # 创建 Agent
    print("步骤 2: 创建 LangGraph Agent...")
    agent_executor = create_sql_agent()
    print("✓ SQL Agent 使用 LangGraph 初始化成功")
    
except Exception as e:
    print(f"❌ 初始化期间出错: {e}")
    import traceback
    traceback.print_exc()
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
            # 市场分析AI助手
            使用 LangGraph 构建的代理，结合 DeepSeek 和 GLM-4 模型，通过自然语言与 DuckDB 数据库进行交互。
            """
        )
    with gr.Tab("项目简介"):
        gr.Markdown(
            f"""
            ### 功能特性
            - **自然语言查询**: 用户可以使用自然语言提问，系统会自动将其转换为 SQL 查询。
            - **多模型支持**: 意图识别和答案生成使用 GLM-4 模型，SQL 生成和验证使用 DeepSeek 模型。
            - **SQL 验证与修复**: 生成的 SQL 查询会经过验证和可能的修复，以提高查询成功率。
            - **详细的思考过程**: Agent 会展示其思考步骤、生成的 SQL 查询以及查询结果，方便用户理解和调试。
            - **对话历史支持**: (当前版本代码中未明确实现，但 LangGraph 结构支持扩展)
            - **可配置性**: 通过 `.env` 文件轻松配置 API 密钥和模型端点。
            - **LangSmith 集成**: 可选的 LangSmith 集成，用于跟踪和调试 Agent 的运行过程。

            ### 当前配置
            - 📁 **数据库**: `{os.path.basename(DB_FILE)}` (DuckDB)
            - 🤖 **SQL模型**: `DeepSeek` (通过 VolcEngine Ark 访问, 模型 ID: `{os.getenv('deepseek0324', '未配置')}`)
            - 🧠 **分析模型**: `GLM-4-Flash` (用于意图识别和答案生成, 模型 ID: `GLM-4-Flash-250414`)
            - ⚙️ **配置文件**: `{os.path.basename(ENV_PATH)}`
            """
        )

    with gr.Tab("查询访问"):
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
                "在价格配置表中找出2024年20到25万增程（REEV）suv销量前十车型",
                "2024年哪个城市级别的销量最高？",
                "对比小米SU7和智己L6用户年龄百分比分布差异",
            ],
            chatbot=gr.Chatbot(height=450, show_copy_button=True), 
            textbox=gr.Textbox(placeholder="请输入您关于数据库的问题...", container=False, scale=7),
            retry_btn="🔄 重试",
            undo_btn="↩️ 撤销",
            clear_btn="🗑️ 清除对话",
    )

if __name__ == "__main__":
    print(f"尝试启动 Gradio 界面...")
    demo.launch()