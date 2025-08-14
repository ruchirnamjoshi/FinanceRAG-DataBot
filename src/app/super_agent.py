# src/app/super_agent.py
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag.tool import rag_query_tool
from src.databot.tools import select_csv_tool, run_python_on_current_df,get_current_df_schema

import textwrap

def safe_system(raw: str) -> str:
    # Escape all braces so ChatPromptTemplate won't treat them as variables
    s = raw.replace("{", "{{").replace("}", "}}")
    # Re-enable the few placeholders we actually want
    for keep in ("{tools}", "{tool_names}", "{input}", "{agent_scratchpad}"):
        s = s.replace(keep.replace("{","{{").replace("}","}}"), keep)
    return s

RAW_SYSTEM = """
You are a financial analysis assistant.

Routing:
- For filing Q&A use `rag_query`.
- For charts:
  1) Call `select_csv` (infer ticker/form/table_type/year).
  2) Call `get_current_df_schema` to see exact columns.
  3) Call `run_python_on_current_df`  `df` as the dataframe, return stdout for any plot.Do not import any library, you have df, pd,plt already. Do not add "plt.show()". Make sure you always write code to include both x and y axis.
  4) Do NOT read files; use the provided df.

IMPORTANT:
- If the request contains plot/chart/graph/line/bar/area/visualize/trend/stacked, you MUST call `select_csv` AND `run_python_on_current_df` (and typically `get_current_df_schema`) before Final Answer.
- Never stop after select_csv for chart requests.

Example ReAct:

Question: Plot Apple segment revenue (iPhone vs Services) for 2025 as a bar chart.
Thought: I need to select the CSV, inspect columns, then plot.
Action: select_csv
Action Input: { "ticker":"AAPL", "table_type":"segment_revenue", "year":2025 }
Observation: { "selected":"data/csv_clean/AAPL/2025_Q2_10-Q_segment_revenue_18.csv", "schema":{"columns":["Period","iPhone","Services"], "label_col":"Period"} }
Thought: I should confirm the schema.
Action: get_current_df_schema
Action Input: {}
Observation: { "columns":["Period","iPhone","Services"], "label_col":"Period", "numeric_cols":["iPhone","Services"] }
Thought: Now I can plot with the correct column names.
Action: run_python_on_current_df
Action Input: { "code":"df.plot(kind='bar', x='Quarter', y=['iPhone']); plt.xlabel('Quarter'); plt.ylabel('Revenue (Billions)')" }
Observation: { "image_b64":"..." }
Thought: I now know the final answer.
Final Answer: Here is the requested chart.
"""

SYSTEM = safe_system(textwrap.dedent(RAW_SYSTEM))

TOOLS = [rag_query_tool, select_csv_tool, run_python_on_current_df, get_current_df_schema]

def _llm(provider="ollama"):
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
    return ChatOllama(model=os.getenv("LLM_MODEL","llama3.1:8b"), temperature=0)

def build_super_agent(provider="ollama") -> AgentExecutor:
    llm = _llm(provider)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         SYSTEM
         + "\n\nYou have access to the following tools:\n{tools}\n"
         "When you use a tool, reference it by name from: {tool_names}."),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}"),
        ("ai", "{agent_scratchpad}"),
    ])
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
    # return_intermediate_steps=True -> we can inspect steps in the UI
    return AgentExecutor(
        agent=agent, tools=TOOLS, verbose=True,
        handle_parsing_errors=True, max_iterations=6,
        return_intermediate_steps=True
    )