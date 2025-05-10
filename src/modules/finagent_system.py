from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
import pytz
from uuid import uuid4

# Langchain và Qdrant imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import các module agent
from modules.low_level import agent as technical_analysis_agent
from modules.high_level import agent as high_level_agent

from modules.decision import agent as decision_agent

from modules.market_intelligence.graph import run_market_intelligence_agent
from modules.risk_manager.nodes import run_risk_manager_agent
from modules.trading_strategy.agents.buffett import run_buffett_analysis
from modules.trading_strategy.agents.lynch import  run_lynch_analysis
from modules.trading_strategy.agents.graham import run_graham_analysis
from modules.trading_strategy.agents.murphy import run_murphy_analysis
from modules.risk_manager.state import TradeDecision



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
    holding: Optional[Dict[str, int]]         # Từ Market Intelligent Agent
    technical_analysis: Optional[str]     # Từ Technical Analysis Agent
    improvement_data: Optional[str]       # Từ High-Level Reflection Agent
    trading_strategy: Optional[str]       # Từ Strategy Agent
    risk_assessment: Optional[str]        # Từ Risk Management Agent
    decision: Optional[TradeDecision]         # Từ Decision Agent
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Các Node của Master Workflow ---

def market_intelligent_node(state: MasterState) -> MasterState:
    """Node: Market Intelligent Agent - Thu thập và tóm tắt thông tin thị trường."""
    symbol = state["symbol"]
    messages = state["messages"]

    past_analysis, late_analysis = run_market_intelligence_agent(symbol)

    market_state =  ""

    for analysis in past_analysis:
        market_state += f"Phân tích quá khứ: {analysis['analysis']}\n"
        market_state += f"Tóm tắt: {analysis['summaries']}\n"

    for analysis in late_analysis:
        market_state += f"Phân tích mới nhất: {analysis['analysis']}\n"
        market_state += f"Tóm tắt: {analysis['summaries']}\n"



    return {
        **state,
        "market_data": market_state,
        "messages": messages + AIMessage(content=str(market_state))
    }

def technical_analysis_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    messages = state["messages"]

    tech_state = technical_analysis_agent(symbol, market_data, max_reflections=1)

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

    reflection_state = high_level_agent(symbol, market_data, technical_analysis, max_reflections=1)
    
    return {
        **state,
        "improvement_data": reflection_state,
        "messages": messages + reflection_state
    }

def strategy_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    messages = state["messages"]

    buffett_analysis = run_buffett_analysis(symbol)
    lynch_analysis = run_lynch_analysis(symbol)
    graham_analysis = run_graham_analysis(symbol)
    murphy_analysis = run_murphy_analysis(symbol)

    all_trading_strategies = {
        "buffett": buffett_analysis,
        "lynch": lynch_analysis,
        "graham": graham_analysis,
        "murphy": murphy_analysis
    }

    strategy_state = ""
    for strategy, analysis in all_trading_strategies.items():
        strategy_state += f"Phân tích {strategy}:\n"
        strategy_state += f"Signal: {analysis['signal']}\n"
        strategy_state += f"Confidence: {analysis['confidence']}\n"
        strategy_state += f"Reasoning: {analysis['reasoning']}\n"
        




    return {
        **state,
        "trading_strategy": strategy_state,
        "messages": messages + AIMessage(content=str(strategy_state))
    }

def risk_management_node(state: MasterState) -> MasterState:
    
    decision = state["decision"]
    messages = state["messages"]
    holding = state["holding"]

    risk_reduce = run_risk_manager_agent(decision, holding)

    return {
        **state,
        "risk_assessment": risk_reduce[1],
        "decision": risk_reduce[0],
        "messages": messages + AIMessage(content=str(risk_reduce[1]))
    }

def decision_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]
    risk_assessment = state["risk_assessment"]
    messages = state["messages"]

    decision_state = decision_agent(symbol, market_data, technical_analysis, improvement_data, trading_strategy, risk_assessment)

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

    # workflow.add_node("market_intelligent", market_intelligent_node)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("high_level_reflection", high_level_reflection_node)
    # workflow.add_node("strategy", strategy_node)
    # workflow.add_node("risk_management", risk_management_node)
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