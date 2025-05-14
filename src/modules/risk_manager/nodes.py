import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager.state import TradeDecision, RiskLevel, RiskFactor, RiskAssessment, RiskProfile, RiskManagerState
from modules.risk_manager.api import get_concentration_data, get_liquidity_data, get_counterparty_data, get_volatility_data
from typing import Dict, Any

from modules.risk_manager.prompt import get_risk_reduction_prompt
from modules.risk_manager.helper import RiskManagerParser
from modules.risk_manager.tools import compile_risk_profile
from modules.risk_manager.plan import generate_mitigation_plan
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph , END
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def initialize_state(state: RiskManagerState) -> RiskManagerState:
    trade_decision = state.trade_decision


    
    if trade_decision.symbol == "":
        raise ValueError("Symbol is required.")

    if trade_decision.action not in ["BUY", "SELL", "HOLD"]:
        raise ValueError(f"Invalid action: {trade_decision.action}. Must be 'buy', 'sell', or 'hold'.")

    if trade_decision.quantity <= 0:
        raise ValueError("Quantity must be greater than 0.")
    
    if trade_decision.price <= 0:
        raise ValueError("Price must be greater than 0.")

    # generate sample data for testing 
    if trade_decision.exchange_name == "" or trade_decision.exchange_name is None:
        trade_decision.exchange_name = "NASDAQ"

    if state.holding is None:
        state.holding = {
            "AAPL" : 10, 
            "MSFT" : 5,
            "GOOGL" : 2
        }

    state.message.append("Passed initial validation checks.")
    state.next_step = "assess_risk"

    return state

def assess_risk(state: RiskManagerState) -> RiskManagerState: 

    trade_decision = state.trade_decision
    
    vol_data = get_volatility_data(trade_decision.symbol)
    li_data = get_liquidity_data(trade_decision.symbol)
    cp_data = get_counterparty_data(trade_decision.exchange_name)
    conc_data = get_concentration_data(state.holding)

    try : 
        risk_profile = compile_risk_profile(trade_decision, vol_data, li_data, cp_data, conc_data)

        state.risk_profile = risk_profile
        state.message.append("Risk profile compiled successfully.")
        state.next_step = "create_mitigation_plan"

        state.message.append("Risk profile compiled successfully.")
    except Exception as e:
        state.message.append(f"Error compiling risk profile: {e}")
        state.next_step = "error"

    return state



def create_mitigation_plan(state: RiskManagerState) -> RiskManagerState:

    if state.risk_profile is None:
        raise ValueError("Risk profile is required to create a mitigation plan.")

    try:
        mitigation_plan = generate_mitigation_plan(state.trade_decision, state.risk_profile)
        state.mitigation_plan = mitigation_plan
        state.next_step = "generate_final_decision"
        state.message.append("Mitigation plan created successfully.")

    except Exception as e:
        state.message.append(f"Error creating mitigation plan: {e}")
        state.next_step = "error"

    return state


def generate_final_decision(state: RiskManagerState) -> RiskManagerState:

    prompt = get_risk_reduction_prompt(
        trade_decision=state.trade_decision,
        risk_profile=state.risk_profile,
        mitigation_plan=state.mitigation_plan
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        api_key = GEMINI_API_KEY
    )

    chain = llm | RiskManagerParser()

    output = chain.invoke(prompt)
    state.message.append("Final decision generated successfully.")

    state.trade_decision = output.trade_decision
    state.reasoning = output.reasoning
    state.message.append("Final decision generated successfully.")

    state.next_step = "complete"
    return state 


def error_handler(state: RiskManagerState) -> RiskManagerState:
    state.message.append("An error occurred during the risk management process.")
    state.next_step = "error"
    return state


def router(state: RiskManagerState) -> str:
    
    if state.next_step == "error": 
        return "error_handler"
    
    return state.next_step


def create_risk_manager_agent(): 

    workflow = StateGraph(RiskManagerState)

    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("create_mitigation_plan", create_mitigation_plan)
    workflow.add_node("generate_final_decision", generate_final_decision)
    workflow.add_node("error_handler", error_handler)

    workflow.add_edge("initialize_state", "assess_risk")
  
    
    workflow.add_conditional_edges(
       "assess_risk",
       router, 
       {
           "create_mitigation_plan": "create_mitigation_plan",
           "error_handler": "error_handler"
       }
    )

    workflow.add_conditional_edges(
        "create_mitigation_plan",
        router,
        {
            "generate_final_decision": "generate_final_decision",
            "error_handler": "error_handler"
        }
    )

    workflow.add_edge("generate_final_decision", END)
    workflow.add_edge("error_handler", END)
    

    workflow.set_entry_point("initialize_state")
    return workflow.compile()




def run_risk_manager_agent(trade_decision: TradeDecision, holding: Dict[str, int]) -> TradeDecision:

    agent = create_risk_manager_agent()
    state = RiskManagerState(
        trade_decision=trade_decision,
        holding=holding,
    )


    result = agent.invoke(state)

    return result.get("trade_decision", None), result.get("reasoning", None)


if __name__ == "__main__":

    trade_decision = TradeDecision(
        symbol="AAPL",
        action="buy",
        quantity=10,
        price=150.0,
        exchange_name="NASDAQ"
    )

    holding = {
        "AAPL": 10,
        "MSFT": 5,
        "GOOGL": 2
    }

    result = run_risk_manager_agent(trade_decision, holding)

    
    print(result)


graph = create_risk_manager_agent()


