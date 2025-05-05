import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from agent.trading_strategy.state import BuffettState, TradingSignal
from typing import Dict
from datetime import datetime
from agent.trading_strategy.tools import analyze_consistency, analyze_fundamentals, analyze_management_quality, analyze_moat, calculate_intrinsic_value, get_financial_metrics, search_line_items
from agent.trading_strategy.helper import TradingSignalParser

# LangGraph imports
from langgraph.graph import StateGraph, END
import json 
from agent.trading_strategy.prompt.buffett import get_buffett_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




# define nodes
def initialize_state(state: BuffettState) -> BuffettState:
    """Initialize state with ticker and end date."""

    if state.ticker is None: 
        raise ValueError("Ticker is required to initialize the state.")


    ticker = state.ticker
    end_date = state.end_date or datetime.now().strftime("%Y-%m-%d")
    
    
    state.messages.append([
            {"role": "system", "content": "Processing investment analysis for Warren Buffett strategy."},
            {"role": "user", "content": f"Analyzing {ticker} as of {end_date}"}
        ])
    
    state.current_step = "fetch_financial_data"
    return state

def fetch_financial_data(state: BuffettState) -> BuffettState:
    """Fetch all required financial data for analysis."""
    ticker = state.ticker
    end_date = state.end_date
    
    try:
        # Determine paths to potential JSON data files
        current_dir = os.path.dirname(os.path.abspath(__file__))  # trading_strategy folder
        api_dir = os.path.join(os.path.dirname(current_dir), "api")  # api folder
        
        metrics_file = os.path.join(api_dir, f"{ticker}_financial_metrics.json")
        line_items_file = os.path.join(api_dir, f"{ticker}_line_items.json")
        
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
        line_items_to_search = [
            "capital_expenditure",
            "depreciation_and_amortization",
            "net_income",
            "outstanding_shares",
            "total_assets",
            "total_liabilities",
            "dividends_and_other_cash_distributions",
            "issuance_or_purchase_of_equity_shares",
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
        
        # Get market cap
        market_cap = metrics[0].get("market_cap", 0) if metrics else 0
        
        # Update state
        state.metrics = metrics
        state.financial_line_items = financial_line_items
        state.current_step = "analyze_financials"
        state.market_cap = market_cap
        
        # Add a message
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

def analyze_financials(state: BuffettState) -> BuffettState:
    """Run all financial analyses for Buffett strategy."""
   
    
    # Skip if error
    if state.error:
        return state
    
    try:
        # Run all analyses
        metrics = state.metrics
        financial_line_items = state.financial_line_items
        
        # Analyze fundamentals
        fundamental_analysis = analyze_fundamentals(metrics)
        
        # Analyze consistency
        consistency_analysis = analyze_consistency(financial_line_items)
        
        # Analyze moat
        moat_analysis = analyze_moat(metrics)
        
        # Analyze management quality
        mgmt_analysis = analyze_management_quality(financial_line_items)
        
        # Calculate intrinsic value
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)
        
        # Calculate total score
        total_score = (
            fundamental_analysis.get("score", 0) + 
            consistency_analysis.get("score", 0) + 
            moat_analysis.get("score", 0) + 
            mgmt_analysis.get("score", 0)
        )
        
        max_score = 10 + moat_analysis.get("max_score", 3) + mgmt_analysis.get("max_score", 2)
        
        # Calculate margin of safety
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis.get("intrinsic_value")
        market_cap = state.market_cap
        
        if intrinsic_value and market_cap and market_cap > 0:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
        
        # Update state
        state.fundamental_analysis = fundamental_analysis
        state.consistency_analysis = consistency_analysis
        state.moat_analysis = moat_analysis
        state.management_analysis = mgmt_analysis
        state.intrinsic_value_analysis = intrinsic_value_analysis
        state.total_score = total_score
        state.max_score = max_score
        state.margin_of_safety = margin_of_safety
        state.current_step = "generate_signal"
        
        # Add a message
        state.messages.append({
            "role": "assistant", 
            "content": f"Completed financial analysis for {state.ticker}"
        })
        
    except Exception as e:
        state.error = f"Error analyzing financials: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error analyzing financials: {str(e)}"
        })
    
    return state

def generate_signal(buffett_state: BuffettState) -> BuffettState:
    """Generate final Buffett investment signal."""

    
    # Skip if error
    if buffett_state.error:
        return buffett_state.model_dump()
    
    try:
        # Prepare analysis data for LLM
        analysis_data = {
            "ticker": buffett_state.ticker,
            "fundamental_analysis": buffett_state.fundamental_analysis,
            "consistency_analysis": buffett_state.consistency_analysis,
            "moat_analysis": buffett_state.moat_analysis,
            "management_analysis": buffett_state.management_analysis,
            "intrinsic_value_analysis": buffett_state.intrinsic_value_analysis,
            "market_cap": buffett_state.market_cap,
            "margin_of_safety": buffett_state.margin_of_safety,
            "total_score": buffett_state.total_score,
            "max_score": buffett_state.max_score
        }
        
        # Generate preliminary signal based on rules
        total_score = buffett_state.total_score
        max_score = buffett_state.max_score
        margin_of_safety = buffett_state.margin_of_safety
        
        if (total_score >= 0.7 * max_score) and margin_of_safety and (margin_of_safety >= 0.3):
            signal = "bullish"
        elif total_score <= 0.3 * max_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            signal = "bearish"
        else:
            signal = "neutral"
        
        # LLM prompt for Buffett reasoning and refinement
        prompt = get_buffett_prompt(
            ticker=buffett_state.ticker,
            analysis_data=json.dumps(analysis_data, indent=2),
            signal=signal
        )


        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key = GEMINI_API_KEY
        )
        
        # Get LLM response and parse it
        chain = llm | TradingSignalParser()
        buffett_output = chain.invoke(prompt)
    
        # Update state
        buffett_state.output_signal = buffett_output
        buffett_state.current_step = "complete"
        
        # Add final message
        buffett_state.messages.append({
            "role": "assistant", 
            "content": f"Warren Buffett analysis complete for {buffett_state.ticker}: {buffett_output.signal.upper()} with {buffett_output.confidence:.0%} confidence"
        })
        
    except Exception as e:
        buffett_state.error = f"Error generating signal: {str(e)}"
        buffett_state.current_step = "error"
        buffett_state.messages.append({
            "role": "assistant", 
            "content": f"Error generating signal: {str(e)}"
        })
    
    return buffett_state.model_dump()


def handle_error(buffett_state: BuffettState) -> BuffettState:

    
    # Add error message if not already present
    if not any(msg.get("content", "").startswith("Error:") for msg in buffett_state.messages):
        buffett_state.messages.append({
            "role": "assistant", 
            "content": f"Error: {buffett_state.error}"
        })
    
    # Set a default signal in case of error
    buffett_state.output_signal = TradingSignal(
        signal="neutral",
        confidence=0.0,
        reasoning=f"Error occurred during analysis: {buffett_state.error}"
    )
    
    buffett_state.current_step = "complete"
    
    return buffett_state




## RUN WORKFLOW + ROUTER
# Router function to determine next step
def router(buffett_state: BuffettState) -> str:
    """Determines the next node in the graph based on current state."""

    
    if buffett_state.error:
        return "error_handler"
    
    return buffett_state.current_step





# Create Warren Buffett agent graph
def create_buffett_agent() -> StateGraph:
    """Create and return the Warren Buffett agent workflow."""
    # Create a new graph
    workflow = StateGraph(BuffettState)
    
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

# Function to run the agent with a ticker
def run_buffett_analysis(ticker: str, end_date: str = None) -> Dict:
    """Run Warren Buffett analysis for a given ticker."""

    buffett_agent = create_buffett_agent()
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    initial_state = {"ticker": ticker, "end_date": end_date}
    
    # Run the agent
    final_state = buffett_agent.invoke(initial_state)
    
    # Return the final state
    return final_state




if __name__ == "__main__":
    from pprint import pprint
    ticker = "AAPL"
    end_date = "2023-12-31"
    
    # Run the analysis
    result = run_buffett_analysis(ticker, end_date)
    
    # Print the result
    pprint(result)