import os
import gradio as gr
import sqlite3
from dotenv import load_dotenv
import time # Added import

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# --- Configuration ---
ENV_PATH = "/Users/zihao_/Documents/coding/Langchain_chatwithdata/W20方向/.env"
DB_FILE = "chinook_agent.db" # Will be created in the same directory as the script

# --- Load Environment Variables and Configure LangSmith ---
load_dotenv(dotenv_path=ENV_PATH)

def configure_langsmith():
    if os.getenv("ENABLE_LANGSMITH", "false").lower() == "true":
        if os.getenv("LangSmith_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LangSmith_API_KEY")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "SQL_Agent_DeepSeek_Gradio")
            print(f"LangSmith tracing enabled for project: {os.environ['LANGCHAIN_PROJECT']}")
            if not os.getenv("LANGCHAIN_API_KEY"): # Double check after assignment
                 print("Warning: ENABLE_LANGSMITH is true, but LangSmith_API_KEY could not be set for LANGCHAIN_API_KEY. Tracing might not work.")
        else:
            print("Warning: ENABLE_LANGSMITH is true, but LangSmith_API_KEY is not found in .env. Tracing disabled.")
    else:
        print("LangSmith tracing disabled.")

configure_langsmith()

# --- LLM Initialization ---
def get_llm():
    ark_api_key = os.getenv("ARK_API_KEY")
    # The model name/endpoint ID for DeepSeek0324 from your .env file
    model_name_from_env = os.getenv("deepseek0324") 

    if not ark_api_key:
        raise ValueError("ARK_API_KEY must be set in the .env file.")
    if not model_name_from_env:
        # Fallback or error if deepseek0324 is not in .env, though user specified it is.
        print("Warning: 'deepseek0324' not found in .env, using default 'Deepseek-V3'. This might not be what you intend.")
        model_name_from_env = "Deepseek-V3" # A generic fallback, might not work if endpoint ID is required

    return ChatOpenAI(
        openai_api_key=ark_api_key,
        openai_api_base="https://ark.cn-beijing.volces.com/api/v3", # Standard VolcEngine Ark base URL
        model_name=model_name_from_env, # Use the value from .env
        temperature=0
    )

# --- Database Setup ---
def setup_database(db_file=DB_FILE):
    db_path = os.path.join(os.path.dirname(__file__), db_file) # Place DB next to script
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a simple table (e.g., Employees from Chinook example)
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
    # Add some sample data if the table is empty
    cursor.execute("SELECT COUNT(*) FROM Employees")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            (1, 'Adams', 'Andrew', 'General Manager', None, '1962-02-18', '2002-08-14', '11120 Jasper Ave NW', 'Edmonton', 'AB', 'Canada', 'T5K 2N1', '+1 (780) 428-9482', '+1 (780) 428-3457', 'andrew@chinookcorp.com'),
            (2, 'Edwards', 'Nancy', 'Sales Manager', 1, '1958-12-08', '2002-05-01', '825 8 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 2T3', '+1 (403) 262-3443', '+1 (403) 262-3322', 'nancy@chinookcorp.com'),
            (3, 'Peacock', 'Jane', 'Sales Support Agent', 2, '1973-08-29', '2002-04-01', '1111 6 Ave SW', 'Calgary', 'AB', 'Canada', 'T2P 5M5', '+1 (403) 262-3443', '+1 (403) 262-6712', 'jane@chinookcorp.com')
        ]
        cursor.executemany("INSERT INTO Employees VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", sample_data)
        print(f"Inserted sample data into {db_file}.")
    
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"

# --- Initialize LLM, Database, and Agent ---
try:
    llm = get_llm()
    db_uri = setup_database()
    db = SQLDatabase.from_uri(db_uri)

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools", # Recommended for models supporting OpenAI-style tool calls
        verbose=True,
        handle_parsing_errors=True, # Helps with minor LLM output formatting issues
        agent_executor_kwargs={"return_intermediate_steps": True} # To capture SQL queries
    )
    print("SQL Agent initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    # Provide a dummy agent_executor if initialization fails, so Gradio can still load
    agent_executor = None 

# --- Gradio Interaction Function ---
def query_agent(message: str, history: list): # Modified signature
    if agent_executor is None:
        yield "Error: SQL Agent failed to initialize. Please check your .env configuration and console logs."
        return 
    
    if not message.strip(): 
        yield "Please enter a question."
        return

    try:
        response = agent_executor.invoke({"input": message}) 
        
        final_answer = response.get("output", "No final answer found.")
        
        intermediate_steps = response.get("intermediate_steps", [])
        sql_queries_log_entries = []

        for agent_action, observation in intermediate_steps:
            if hasattr(agent_action, 'tool') and agent_action.tool == "sql_db_query":
                query = agent_action.tool_input
                if isinstance(query, dict) and 'query' in query:
                    sql_queries_log_entries.append(f"**SQL Query:**\n```sql\n{query['query']}\n```\n**Result:**\n```\n{observation}\n```\n")
                elif isinstance(query, str):
                    sql_queries_log_entries.append(f"**SQL Query:**\n```sql\n{query}\n```\n**Result:**\n```\n{observation}\n```\n")
            elif hasattr(agent_action, 'tool'): 
                 sql_queries_log_entries.append(f"**Tool:** {agent_action.tool}\n**Input:** {agent_action.tool_input}\n**Observation:** {observation}\n")

        # Construct the full response string first
        full_response_str = final_answer

        if sql_queries_log_entries:
            details_content = "\n".join(sql_queries_log_entries)
            details_html = f"\n\n<details>\n<summary><strong>--- Agent Steps and SQL Queries (Click to expand) ---</strong></summary>\n\n{details_content}\n</details>\n"
            full_response_str += details_html
            
        if not full_response_str.strip():
            yield "Agent did not produce a response."
            return

        # Stream the full response string character by character
        buffer = ""
        for char_token in full_response_str:
            buffer += char_token
            time.sleep(0.01)  # Adjust for typing speed, 0.01 is faster
            yield buffer

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during agent invocation: {error_details}") 
        error_message = (f"An error occurred: {str(e)}\n\n"
                         "Details:\n"
                         f"{error_details}\n\n"
                         "Troubleshooting tips:\n"
                         "- Ensure your .env file has correct ARK_API_KEY and deepseek0324 model ID.\n"
                         "- Verify the DeepSeek model version supports tool usage (function calling).\n"
                         "- Check console logs for more specific error messages from Langchain or the LLM API.")
        yield error_message # Yield the full error message at once

# --- Gradio Interface ---
# Use gr.Blocks for more control over layout and to include Markdown for title/description
with gr.Blocks(theme=gr.themes.Soft()) as demo: # Added theme for a slightly nicer look
    gr.Markdown(
        """
        # Langchain SQL Agent with DeepSeek (via VolcEngine)
        使用 Langchain SQL Agent 和 DeepSeek 模型查询 SQLite 数据库 (Employees表示例).
        """
    )
    gr.Markdown(
        f"""
        LLM 配置从 '{os.path.basename(ENV_PATH)}' 加载. 
        数据库文件: '{DB_FILE}'.
        Agent 会显示最终答案以及可折叠的SQL查询和中间步骤。
        """
    )
    
    chat_interface = gr.ChatInterface(
        fn=query_agent,
        examples=[ 
            "Describe the Employees table",
            "How many employees are there in Canada?",
            "Who are the employees in Calgary?",
            "List all employees",
            "What are the distinct countries in the Employees table?"
        ],
        chatbot=gr.Chatbot(height=500, show_copy_button=True), 
        textbox=gr.Textbox(placeholder="请输入您关于数据库的问题...", container=False, scale=7),
        retry_btn="🔄 重试",
        undo_btn="↩️ 撤销",
        clear_btn="🗑️ 清除对话",
        # flagging_mode="manual",  # Removed due to version incompatibility
        # flagging_options=["👍 喜欢", "👎 不喜欢", "⚠️ 不准确", "其他"], # Removed
    )

if __name__ == "__main__":
    print(f"Attempting to launch Gradio interface...")
    demo.launch() # Changed from iface.launch()