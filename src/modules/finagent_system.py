from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
import pytz
from uuid import uuid4

# Langchain và Qdrant imports
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import các module agent
from modules.market import agent as market_intelligent_agent
from modules.low_level import agent as technical_analysis_agent
from modules.high_level import agent as high_level_agent
from modules.strategy import agent as strategy_agent
from modules.risk_management import agent as risk_management_agent
from modules.decision import agent as decision_agent

# --- Khởi tạo LLM và Embeddings ---
# llm = ChatOllama(model="cogito:3b")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Định nghĩa State cho Master Workflow ---
class MasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_data: Optional[Dict]           # Từ Market Intelligent Agent
    technical_analysis: Optional[str]     # Từ Technical Analysis Agent
    improvement_data: Optional[str]       # Từ High-Level Reflection Agent
    trading_strategy: Optional[str]       # Từ Strategy Agent
    risk_assessment: Optional[str]        # Từ Risk Management Agent
    final_decision: Optional[str]         # Từ Decision Agent
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Các Node của Master Workflow ---

def market_intelligent_node(state: MasterState) -> MasterState:
    """Node: Market Intelligent Agent - Thu thập và tóm tắt thông tin thị trường."""
    symbol = state["symbol"]
    messages = state["messages"]

    market_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": None,
        "market_news": None,
        "market_context": None,
        "market_summary": None,
        "history_query": None,
        "past_market": None,
        "final_output": None,
    }

    market_state = market_intelligent_agent(market_state)

    return {
        **state,
        "market_data": market_state,
        "messages": market_state["messages"]
    }

def technical_analysis_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    messages = state["messages"]

    tech_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data,
        "technical_indicators": None,
        "technical_analysis": None,
        "reflection_query": None,
        "historical_insights": None,
        "reflection_iteration": 0,
        "max_reflections": 1,
        "final_output": None
    }

    tech_state = technical_analysis_agent(tech_state)

    return {
        **state,
        "technical_analysis": tech_state,
        "messages": messages + tech_state
    }

def high_level_reflection_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    messages = state["messages"]

    reflection_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data,
        "technical_analysis": technical_analysis,
        "trade_history": None,
        "reflection_query": None,
        "historical_insights": None,
        "improvement_suggestions": None,
        "reflection_iteration": 0,
        "max_reflections": 1,
        "final_output": None
    }


    reflection_state = high_level_agent(reflection_state)
    
    return {
        **state,
        "improvement_data": reflection_state,
        "messages": messages + reflection_state
    }

def strategy_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    improvement_data = state["improvement_data"]
    messages = state["messages"]

    strategy_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data,
        "research_data": technical_analysis,
        "improvement_data": improvement_data,
        "trading_strategy": None,
        "final_output": None
    }

    strategy_state = strategy_agent(strategy_state)

    return {
        **state,
        "trading_strategy": strategy_state,
        "messages": messages + strategy_state
    }

def risk_management_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]
    messages = state["messages"]


    risk_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data["raw_data"],
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
        "risk_assessment": None,
        "final_output": None
    }

    risk_state = risk_management_agent(risk_state)

    return {
        **state,
        "risk_assessment": risk_state,
        "messages": messages + risk_state
    }

def decision_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]
    risk_assessment = state["risk_assessment"]
    messages = state["messages"]

    decision_state = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data["raw_data"],
        "technical_analysis": technical_analysis,
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
        "risk_assessment": risk_assessment,
        "final_decision": None,
        "final_output": None
    }

    decision_state = decision_agent(decision_state)

    return {
        **state,
        "final_decision": decision_state,
        "messages": messages + decision_state
    }

def format_final_output(state: MasterState) -> MasterState:
    """Node: Định dạng kết quả cuối cùng."""
    final_decision = state["final_decision"]
    messages = state["messages"]



    output = f"""{final_decision}"""

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Master Workflow ---

def build_master_workflow():
    """Xây dựng và biên dịch Master Workflow."""
    workflow = StateGraph(MasterState)

    workflow.add_node("market_intelligent", market_intelligent_node)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("high_level_reflection", high_level_reflection_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("risk_management", risk_management_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "market_intelligent")
    workflow.add_edge("market_intelligent", "technical_analysis")
    workflow.add_edge("technical_analysis", "high_level_reflection")
    workflow.add_edge("high_level_reflection", "strategy")
    workflow.add_edge("strategy", "risk_management")
    workflow.add_edge("risk_management", "decision")
    workflow.add_edge("decision", "format_final_output")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"

    graph = build_master_workflow()

    initial_state = {
        "messages": [HumanMessage(content=f"Thực hiện quy trình giao dịch cho {symbol_to_analyze}.")],
        "symbol": symbol_to_analyze,
        "market_data": None,
        "technical_analysis": None,
        "improvement_data": None,
        "trading_strategy": None,
        "risk_assessment": None,
        "final_decision": None,
        "final_output": None,
    }

    print(f"\n--- Bắt đầu Master Workflow cho {symbol_to_analyze} ---")
    final_result_state = graph.invoke(initial_state)
    print(f"\n--- Master Workflow cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Cuối cùng ---")
    if final_result_state.get("final_output"):
        print(final_result_state["final_output"])
    else:
        print("Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")
        if final_result_state.get("messages"):
            last_message = final_result_state["messages"][-1].content
            if "Lỗi:" in last_message:
                print("\nLỗi gặp phải:", last_message)