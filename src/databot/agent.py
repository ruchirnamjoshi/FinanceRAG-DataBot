# src/databot/agent.py
import os
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

from .tools import ingest_csv_tool, summarize_csv_tool, select_csv_tool

TOOLS = [ingest_csv_tool, select_csv_tool, summarize_csv_tool]

SYSTEM = (
    "You are DataBot, an expert data assistant. "
    "Always select the most relevant CSV (use select_csv) before summarizing or executing code, "
    "unless a CSV was just ingested or the user explicitly names a file. "
    "When plotting or transforming data, call execute_code with Python using df."
)

def build_llm(provider: Literal["openai","ollama"]="ollama"):
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), temperature=0)
    return ChatOllama(model=os.getenv("LLM_MODEL","llama3.1:8b"), temperature=0)

def build_agent(provider: Literal["openai","ollama"]="ollama"):
    llm = build_llm(provider)
    tool_text = render_text_description(TOOLS)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nAvailable tools:\n" + tool_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_react_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

def get_history(session_id: str):
    from pathlib import Path
    db = os.getenv("CHAT_DB_PATH","data/chat_history.sqlite3")
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    return SQLChatMessageHistory(session_id=session_id, connection_string=f"sqlite:///{db}")

def chat_once(agent: AgentExecutor, user_text: str, session_id="default"):
    hist = get_history(session_id)
    result = agent.invoke({"input": user_text, "chat_history": hist.messages})
    # store
    hist.add_user_message(user_text)
    hist.add_ai_message(result["output"])
    return result["output"]
