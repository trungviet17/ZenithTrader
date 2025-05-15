import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from modules.trading_strategy.state import LynchState, TradingSignal
from typing import Dict
from datetime import datetime
import json
from modules.trading_strategy.helper import TradingSignalParser
from modules.trading_strategy.tools.lynch_tools import (
    analyze_growth,
    analyze_peg_ratio,
    analyze_competitive_advantage,
    get_industry_data
)
from modules.trading_strategy.tools.api_tools import  get_financial_metrics, search_line_items
from modules.trading_strategy.prompt.lynch import get_lynch_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.utils.llm import LLM

# LangGraph imports
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define nodes
def initialize_state(state: LynchState) -> LynchState:
    """Initialize state with ticker and end date."""
    if state.ticker is None: 
        raise ValueError("Ticker is required to initialize the state.")

    ticker = state.ticker
    end_date = state.end_date or datetime.now().strftime("%Y-%m-%d")
    
    state.messages.append([
        {"role": "system", "content": "Processing investment analysis using Peter Lynch's growth at reasonable price strategy."},
        {"role": "user", "content": f"Analyzing {ticker} as of {end_date}"}
    ])
    
    state.current_step = "fetch_financial_data"
    return state

def fetch_financial_data(state: LynchState) -> LynchState:
    """Fetch all required financial data for Lynch analysis."""
    ticker = state.ticker
    end_date = state.end_date
    
    try:
        # Determine paths to potential JSON data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.join(os.path.dirname(current_dir), "api")
        
        metrics_file = os.path.join(api_dir, f"{ticker}_financial_metrics.json")
        line_items_file = os.path.join(api_dir, f"{ticker}_line_items_lynch.json")
        
        # Check for metrics file and load if exists
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
        
        # Check for line items file and load if exists
        # Lynch focused on growth metrics
        
        
        # Get market cap
        market_cap = metrics[0].get("market_cap", 0) if metrics else 0
        
        # Get industry data (Lynch emphasized understanding the industry)
        industry_data = get_industry_data(ticker)
        
        # Update state
        state.metrics = metrics
    
        state.current_step = "analyze_financials"
        state.market_cap = market_cap
        state.industry_data = industry_data
        
        # Add a message
        state.messages.append({
            "role": "assistant", 
            "content": f"Successfully fetched financial data and industry information for {ticker}"
        })
        
    except Exception as e:
        state.error = f"Error fetching financial data: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error fetching financial data: {str(e)}"
        })
    
    return state



def analyze_financials(state: LynchState) -> LynchState:
    """Run all financial analyses using Lynch's criteria."""
    # Skip if error
    if state.error:
        return state
    
    try:
        # Run all analyses
        metrics = state.metrics
        
        growth_analysis = analyze_growth(metrics)
        
       
        peg_analysis = analyze_peg_ratio(metrics)
        
     
        competitive_advantage = analyze_competitive_advantage(metrics)
        
     
        business_category = growth_analysis.get("category")
        
        
        institutional_ownership = 0.45  
        
      
        total_score = (
            growth_analysis.get("score", 0) + 
            peg_analysis.get("score", 0) + 
            competitive_advantage.get("score", 0)
        )
        
       
        max_score = 15
        
        # Update state
        state.growth_analysis = growth_analysis
        state.peg_analysis = peg_analysis
        state.business_category = business_category
        state.institutional_ownership = institutional_ownership
        state.competitive_advantage = competitive_advantage
        state.total_score = total_score
        state.max_score = max_score
        state.current_step = "generate_signal"
        
        # Add a message
        state.messages.append({
            "role": "assistant", 
            "content": f"Completed financial analysis using Lynch's growth investing principles for {state.ticker}"
        })
        
    except Exception as e:
        state.error = f"Error analyzing financials: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error analyzing financials: {str(e)}"
        })
    
    return state



def generate_signal(state: LynchState) -> Dict:
    """Generate final Lynch investment signal."""
    # Skip if error
    if state.error:
        return state.model_dump()
    
    try:
        # Prepare analysis data for LLM
        analysis_data = {
            "ticker": state.ticker,
            "growth_analysis": state.growth_analysis,
            "peg_analysis": state.peg_analysis,
            "business_category": state.business_category,
            "institutional_ownership": state.institutional_ownership,
            "competitive_advantage": state.competitive_advantage,
            "industry_data": state.industry_data,
            "total_score": state.total_score,
            "max_score": state.max_score,
            "market_cap": state.market_cap
        }
        
        # Generate preliminary signal based on Lynch's approach
        total_score = state.total_score
        max_score = state.max_score
        
        # Lynch was growth-oriented but practical about valuation
        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.4 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Lynch was particularly excited by "fast growers" with good PEG ratios
        if state.business_category == "Fast Grower" and state.peg_analysis.get("score", 0) >= 2:
            signal = "bullish"  # Override if it's a fast grower with good PEG
        
        # LLM prompt for Lynch's reasoning and refinement
        prompt = get_lynch_prompt(
            ticker=state.ticker,
            analysis_data=json.dumps(analysis_data, indent=2),
            signal=signal
        )
        
        llm = LLM.get_gemini_llm(model_index = 1) 
        # Get LLM response and parse it
        chain = llm | TradingSignalParser()
        lynch_output = chain.invoke(prompt)
        
        # Update state
        state.output_signal = lynch_output
        state.current_step = "complete"
        
        # Add final message
        state.messages.append({
            "role": "assistant", 
            "content": f"Peter Lynch analysis complete for {state.ticker}: {lynch_output.signal.upper()} with {lynch_output.confidence:.0%} confidence"
        })
        
    except Exception as e:
        state.error = f"Error generating signal: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error generating signal: {str(e)}"
        })
    
    return state.model_dump()

def handle_error(state: LynchState) -> LynchState:
    # Add error message if not already present
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
def router(state: LynchState) -> str:
    if state.error:
        return "error_handler"
    
    return state.current_step

# Create Peter Lynch agent graph
def create_lynch_agent() :
    """Create and return the Peter Lynch agent workflow."""
    # Create a new graph
    workflow = StateGraph(LynchState)
    
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
    
    # Set the entry point
    workflow.set_entry_point("initialize")
    
    # Compile the graph
    return workflow.compile()


def run_lynch_analysis(ticker: str, end_date: str = None) -> str:
    """Run Warren Buffett analysis for a given ticker."""

    agent = create_lynch_agent()
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    initial_state = {"ticker": ticker, "end_date": end_date}
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    result = final_state.get("output_signal", None)


    output = "Analysis Graham trading strategy\n"
    output += f"Signal {result.get('signal', None)} \n"
    output += f"Confidence {result.get('confidence', None)} \n"
    output += f"Reasoning {result.get('reasoning', None)} \n"



    return output