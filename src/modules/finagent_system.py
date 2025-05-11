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
from modules.low_level.agent import low_level_agent
from modules.high_level.agent import high_level_reflection_agent
from server.schema import AssetData

from modules.market_intelligence.graph import run_market_intelligence_agent
from modules.risk_manager.nodes import run_risk_manager_agent
from modules.trading_strategy.agents.buffett import run_buffett_analysis
from modules.trading_strategy.agents.lynch import  run_lynch_analysis
from modules.trading_strategy.agents.graham import run_graham_analysis
from modules.trading_strategy.agents.murphy import run_murphy_analysis
from modules.risk_manager.state import TradeDecision
from modules.decision.prompt import get_decision_prompt
from modules.decision.helper import DecisionOutputParser
from modules.utils.llm import LLM 
from modules.market_intelligence.tools import MarketSearchingTools


class MasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] = []
    symbol: str
    market_data: Optional[str]           
    holding: Optional[Dict[str, int]]         # Từ Market Intelligent Agent
    technical_analysis: Optional[str]     # Từ Technical Analysis Agent
    improvement_data: Optional[str]       # Từ High-Level Reflection Agent
    trading_strategy: Optional[str]       # Từ Strategy Agent
    risk_assessment: Optional[str]        # Từ Risk Management Agent
    decision: Optional[TradeDecision]         # Từ Decision Agent
    final_output: Optional[str]           # Kết quả cuối cùng
    asset_data: AssetData = None 



def initialize_state(state: MasterState) -> MasterState:

    if state.get("symbol") is None: 
        raise ValueError("Symbol không được cung cấp trong state.")

    if state.get("holding") is None:
        state["holding"] = {
            "AAPL": 10,
            "MSFT": 5,
            "GOOGL": 2
        }

    symbol = state["symbol"]
    asset_data = AssetData(
        asset_symbol='AAPL',
        asset_name='Apple Inc',
        asset_type='Common Stock',
        asset_exchange='NASDAQ',
        asset_sector='TECHNOLOGY',
        asset_industry='ELECTRONIC COMPUTERS',
        asset_description="Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totalling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook."
    )
    state["asset_data"] = asset_data

    state["messages"].append(HumanMessage(content=f"Thực hiện quy trình giao dịch cho {symbol}."))


    return state 



def market_intelligent_node(state: MasterState) -> MasterState:
    
    symbol = state["symbol"]
    asset_data = state["asset_data"]
    messages = state["messages"]

    past_analysis, late_analysis = run_market_intelligence_agent(symbol, asset_data)

    market_state =  ""

    for analysis in past_analysis.analysis:
        market_state += f"Past market intelligence analysis: {analysis}\n"

    market_state += f"Past market intelligence summary: {past_analysis.summaries}\n"

    for analysis in late_analysis.analysis:
        market_state += f"Latest market intelligence analysis: {analysis}\n"
    market_state += f"Latest market intelligence summary: {late_analysis.summaries}\n"



    return {
        **state,
        "market_data": market_state,
    }

def technical_analysis_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    messages = state["messages"]

    tech_state = low_level_agent(symbol, market_data, max_reflections=1)

    return {
        **state,
        "technical_analysis": tech_state,
    }

def high_level_reflection_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    messages = state["messages"]

    sample_trade_history = [
        {
            "trade_id": str(uuid4()),
            "symbol": symbol,
            "date": "2025-02-15",
            "action": "Buy",
            "price": 150.0,
            "quantity": 100,
            "reason": "RSI gives oversold signal, positive MACD crossover.",
            "outcome": "Profit",
            "profit_loss": 500.0,
            "analysis": "Correct decision as the price increased after the technical signal."
        },
        {
            "trade_id": str(uuid4()),
            "symbol": symbol,
            "date": "2025-03-10",
            "action": "Sell",
            "price": 155.0,
            "quantity": 100,
            "reason": "Price hit strong resistance at 155, RSI overbought.",
            "outcome": "Loss",
            "profit_loss": -200.0,
            "analysis": "Wrong decision as the price continued to rise after selling."
        }
    ]

    reflection_state = high_level_reflection_agent(symbol, market_data, technical_analysis, sample_trade_history, max_reflections=1)
    
    return {
        **state,
        "improvement_data": reflection_state,
    }

def strategy_node(state: MasterState) -> MasterState:
    symbol = state["symbol"]
    messages = state["messages"]

    buffett_analysis = run_buffett_analysis(symbol)
    lynch_analysis = run_lynch_analysis(symbol)
    graham_analysis = run_graham_analysis(symbol)
    murphy_analysis = run_murphy_analysis(symbol)


    strategy_state = buffett_analysis + "\n" + lynch_analysis + "\n" + graham_analysis + "\n" + murphy_analysis
    return {
        **state,
        "trading_strategy": strategy_state,
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
    }

def decision_node(state: MasterState) -> MasterState:
    
    llm = LLM.get_gemini_llm()

    chain = llm | DecisionOutputParser()

    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]

    prompt = get_decision_prompt(
        market_intelligence=market_data,
        low_level_reflection=technical_analysis,
        high_level_reflection=improvement_data,
        trading_strategy=trading_strategy, 
        asset_data = state["asset_data"],
    )

    decision = chain.invoke(prompt)
        
    return {
        **state,
        "decision": decision
    }



def build_master_workflow():
    """Xây dựng và biên dịch Master Workflow."""
    workflow = StateGraph(MasterState)
    
    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("market_intelligent", market_intelligent_node)
    workflow.add_node("low_level_reflection", technical_analysis_node)
    workflow.add_node("high_level_reflection", high_level_reflection_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("risk_management", risk_management_node)
    workflow.add_node("trade_decision", decision_node)

    workflow.add_edge(START, "initialize_state")
    workflow.add_edge("initialize_state", "market_intelligent")
    workflow.add_edge("market_intelligent", "low_level_reflection")
    workflow.add_edge("low_level_reflection", "high_level_reflection")
    workflow.add_edge("high_level_reflection", "strategy")
    workflow.add_edge("strategy", "trade_decision")
    workflow.add_edge("trade_decision", "risk_management")
    workflow.add_edge("risk_management", END)

    return workflow.compile()



graph = build_master_workflow()