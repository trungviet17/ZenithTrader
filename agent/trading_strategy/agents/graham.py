import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from agent.trading_strategy.state import GrahamState, TradingSignal
from typing import Dict
from datetime import datetime
import json
from agent.trading_strategy.helper import TradingSignalParser
from agent.trading_strategy.tools.graham_tools import (
    analyze_value_metrics,
    analyze_safety_margin,
    analyze_financial_strength,
    analyze_earnings_stability,
    calculate_graham_intrinsic_value
)
from agent.trading_strategy.tools.api_tools import (
    get_financial_metrics, 
    search_line_items,
)
from agent.trading_strategy.prompt.graham import get_graham_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define nodes
def initialize_state(state: GrahamState) -> GrahamState:
    """Initialize state with ticker and end date."""
    if state.ticker is None: 
        raise ValueError("Ticker is required to initialize the state.")

    ticker = state.ticker
    end_date = state.end_date or datetime.now().strftime("%Y-%m-%d")
    
    state.messages.append([
        {"role": "system", "content": "Processing investment analysis using Benjamin Graham's value investing strategy."},
        {"role": "user", "content": f"Analyzing {ticker} as of {end_date}"}
    ])
    
    state.current_step = "fetch_financial_data"
    return state

def fetch_financial_data(state: GrahamState) -> GrahamState:
    """Fetch all required financial data for Graham analysis."""
    ticker = state.ticker
    end_date = state.end_date
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.join(os.path.dirname(current_dir), "api")
        
        metrics_file = os.path.join(api_dir, f"{ticker}_financial_metrics.json")
        line_items_file = os.path.join(api_dir, f"{ticker}_line_items_graham.json")

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                metrics = metrics_data.get('financial_metrics', [])
                state.messages.append({
                    "role": "assistant", 
                    "content": f"Loaded financial metrics from file for {ticker}"
                })
        else:
            # Fall back to API call
            metrics = get_financial_metrics(ticker, limit=10)['financial_metrics']
            with open(metrics_file, 'w') as f:
                json.dump({"financial_metrics": metrics}, f)
                state.messages.append({
                    "role": "assistant", 
                    "content": f"Fetched financial metrics from API for {ticker}"
                })
        
        line_items_to_search = [
            "outstanding_shares",
            "total_assets",
            "total_liabilities",
            "net_income",
            "retained_earnings",
        ]
        
        if os.path.exists(line_items_file):
            with open(line_items_file, 'r') as f:
                financial_line_items = json.load(f)
                financial_line_items = financial_line_items.get('search_results', [])
                state.messages.append({
                    "role": "assistant", 
                    "content": f"Loaded line items from file for {ticker}"
                })
        else:
            # Fall back to API call
            financial_line_items = search_line_items(
                ticker,
                line_items_to_search,
                end_date,
            )['search_results']

            with open(line_items_file, 'w') as f:
                json.dump({"search_results": financial_line_items}, f)
                state.messages.append({
                    "role": "assistant", 
                    "content": f"Fetched line items from API for {ticker}"
                })
        
        market_cap = metrics[0].get("market_cap", 0) if metrics else 0

        state.metrics = metrics
        state.financial_line_items = financial_line_items
        state.current_step = "analyze_financials"
        state.market_cap = market_cap

        state.messages.append({
            "role": "assistant", 
            "content": f"Successfully fetched financial data for {ticker}"
        })
        
    except Exception as e:
        state.error = f"Error fetching financial data: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error fetching financial data: {str(e)}"
        })
    
    return state

def analyze_financials(state: GrahamState) -> GrahamState:
    """Run all financial analyses using Graham's criteria."""

    if state.error:
        return state
    
    try:
        metrics = state.metrics
        financial_line_items = state.financial_line_items

        value_analysis = analyze_value_metrics(metrics)
        safety_analysis = analyze_safety_margin(metrics, financial_line_items)
        financial_strength_analysis = analyze_financial_strength(metrics)
        earnings_stability_analysis = analyze_earnings_stability(metrics)
        intrinsic_value_result = calculate_graham_intrinsic_value(metrics, financial_line_items)
        intrinsic_value = intrinsic_value_result.get("intrinsic_value")
        
        margin_of_safety = None
        market_cap = state.market_cap
        
        if intrinsic_value and market_cap and market_cap > 0:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
   
        total_score = (
            value_analysis.get("score", 0) + 
            safety_analysis.get("score", 0) + 
            financial_strength_analysis.get("score", 0) + 
            earnings_stability_analysis.get("score", 0)
        )
        
        max_score = 12  # Sum of maximum scores from all analyses
        
        # Update state
        state.value_analysis = value_analysis
        state.safety_analysis = safety_analysis
        state.financial_strength_analysis = financial_strength_analysis
        state.earnings_stability_analysis = earnings_stability_analysis
        state.intrinsic_value = intrinsic_value
        state.margin_of_safety = margin_of_safety
        state.total_score = total_score
        state.max_score = max_score
        state.current_step = "generate_signal"
        
        # Add a message
        state.messages.append({
            "role": "assistant", 
            "content": f"Completed financial analysis using Graham's value investing principles for {state.ticker}"
        })
        
    except Exception as e:
        state.error = f"Error analyzing financials: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error analyzing financials: {str(e)}"
        })
    
    return state

def generate_signal(state: GrahamState) -> Dict:
    """Generate final Graham investment signal."""

    if state.error:
        return state.model_dump()
    
    try:
        analysis_data = {
            "ticker": state.ticker,
            "value_analysis": state.value_analysis,
            "safety_analysis": state.safety_analysis,
            "financial_strength_analysis": state.financial_strength_analysis,
            "earnings_stability_analysis": state.earnings_stability_analysis,
            "intrinsic_value": state.intrinsic_value,
            "market_cap": state.market_cap,
            "margin_of_safety": state.margin_of_safety,
            "total_score": state.total_score,
            "max_score": state.max_score
        }
        
        total_score = state.total_score
        max_score = state.max_score
        margin_of_safety = state.margin_of_safety

        if (total_score >= 0.7 * max_score) and margin_of_safety and (margin_of_safety >= 0.5):
            signal = "bullish"  
        elif total_score <= 0.5 * max_score or (margin_of_safety is not None and margin_of_safety <= 0):
            signal = "bearish"
        else:
            signal = "neutral"
        
        prompt = get_graham_prompt(
            ticker=state.ticker,
            analysis_data=json.dumps(analysis_data, indent=2),
            signal=signal
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=GEMINI_API_KEY
        )
        
        chain = llm | TradingSignalParser()
        graham_output = chain.invoke(prompt)
        
        # Update state
        state.output_signal = graham_output
        state.current_step = "complete"
        
        # Add final message
        state.messages.append({
            "role": "assistant", 
            "content": f"Benjamin Graham analysis complete for {state.ticker}: {graham_output.signal.upper()} with {graham_output.confidence:.0%} confidence"
        })
        
    except Exception as e:
        state.error = f"Error generating signal: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error generating signal: {str(e)}"
        })
    
    return state.model_dump()

def handle_error(state: GrahamState) -> GrahamState:

    if not any(msg.get("content", "").startswith("Error:") for msg in state.messages):
        state.messages.append({
            "role": "assistant", 
            "content": f"Error: {state.error}"
        })
    
    # Set a default signal in case of error
    state.output_signal = TradingSignal(
        signal="neutral",
        confidence=0.0,
        reasoning=f"Error occurred during analysis: {state.error}"
    )
    
    state.current_step = "complete"
    
    return state

# Router function to determine next step
def router(state: GrahamState) -> str:
    if state.error:
        return "error_handler"
    
    return state.current_step

# Create Benjamin Graham agent graph
def create_graham_agent() -> StateGraph:
    """Create and return the Benjamin Graham agent workflow."""
    workflow = StateGraph(GrahamState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("fetch_financial_data", fetch_financial_data)
    workflow.add_node("analyze_financials", analyze_financials)
    workflow.add_node("generate_signal", generate_signal)
    workflow.add_node("error_handler", handle_error)
    
    # Add edges
    workflow.add_edge("initialize", "fetch_financial_data")
    workflow.add_conditional_edges(
        "fetch_financial_data",
        router,
        {
            "error_handler": "error_handler",
            "analyze_financials": "analyze_financials"
        }
    )
    workflow.add_conditional_edges(
        "analyze_financials",
        router,
        {
            "error_handler": "error_handler",
            "generate_signal": "generate_signal"
        }
    )
    workflow.add_edge("generate_signal", END)
    workflow.add_edge("error_handler", END)
    workflow.set_entry_point("initialize")
    
    return workflow.compile()
