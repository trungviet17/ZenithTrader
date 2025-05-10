import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from modules.trading_strategy.state import MurphyState, TradingSignal
from typing import Dict 

import json
import pandas as pd
from modules.trading_strategy.helper import TradingSignalParser
from modules.trading_strategy.tools.murphy_tools import (
    calculate_ma,
    calculate_momentum,
    calculate_sup_res, 
    analyze_volume,
    analyze_trend,
    identify_patterns,
    analyze_momentum
)
from modules.trading_strategy.tools.api_tools import get_history_price

from modules.trading_strategy.prompt.murphy import get_murphy_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


from langgraph.graph import StateGraph, END 
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# define node 

def initialize_state(state: MurphyState) -> MurphyState: 

    if state.ticker is None: 
        raise ValueError("Ticker is required to initialize the state.")
    state.end_date = datetime.now().strftime("%Y-%m-%d")
    ticker = state.ticker
    end_date = state.end_date or datetime.now().strftime("%Y-%m-%d")
    state.interval = "day"
    
 
    
    state.messages.append([
        {"role": "system", "content": "Processing investment analysis using John Murphy's technical analysis approach."},
        {"role": "user", "content": f"Analyzing {ticker} as of {end_date} with {state.interval} "}
    ])
    
    state.current_step = "fetch_price_data"
    return state



def fetch_price_data(state: MurphyState) -> MurphyState: 
    ticker = state.ticker
    interval = state.interval
    start_date = (datetime.now() -   timedelta(days = 100)).strftime("%Y-%m-%d")
    end_date = state.end_date or datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Determine paths to potential JSON data files
        current_dir = os.path.dirname(os.path.abspath(__file__))  # trading_strategy folder
        api_dir = os.path.join(os.path.dirname(current_dir), "api")  # api folder
        
        history_price_file = os.path.join(api_dir, f"{ticker}_history_price.csv")

        if os.path.exists(history_price_file): 
            price_df = pd.read_csv(history_price_file)
        else : 
            price_df = get_history_price(ticker, end_date = state.end_date, start_date= start_date,  interval=interval)
            price_df.to_csv(history_price_file)
        
        if price_df.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        if not isinstance(price_df.index, pd.DatetimeIndex):
            dates_list = price_df.index.tolist()  # Just use numeric indices without formatting
        else:
            dates_list = price_df.index.strftime('%Y-%m-%d').tolist()
        # Store raw price data
        state.price_history = {
            "dates": dates_list,
            "open": price_df['open'].tolist(),
            "high": price_df['high'].tolist(),
            "low": price_df['low'].tolist(),
            "close": price_df['close'].tolist(),
            "volume": price_df['volume'].tolist()
        }
        
        state.current_step = "calculate_technical_indicators"
        state.messages.append({
            "role": "assistant", 
            "content": f"Successfully fetched price history for {ticker} covering {len(price_df)} periods"
        })

    except Exception as e:
        state.error = f"Error fetching price data: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error fetching price data: {str(e)}"
        })
    
    return state


def calculate_technical_indicators(state: MurphyState) -> MurphyState:
    """Calculate technical indicators from price data."""
    # Skip if error
    if state.error:
        return state
    
    try:
        # Convert stored price data back to DataFrame for calculations
        dates = pd.DatetimeIndex(state.price_history['dates'])
        price_df = pd.DataFrame({
            'open': state.price_history['open'],
            'high': state.price_history['high'],
            'low': state.price_history['low'],
            'close': state.price_history['close'],
            'volume': state.price_history['volume']
        }, index=dates)
        
        # Calculate all technical indicators
        price_df = calculate_ma(price_df)
        price_df = calculate_momentum(price_df)
        
        # Store the technical data
        state.technical_data = {
            "last_price": price_df['close'].iloc[-1],
            "sma_20": price_df['SMA_20'].iloc[-1] if 'SMA_20' in price_df else None,
            "sma_50": price_df['SMA_50'].iloc[-1] if 'SMA_50' in price_df else None,
            "sma_200": price_df['SMA_200'].iloc[-1] if 'SMA_200' in price_df else None,
            "rsi": price_df['RSI'].iloc[-1] if 'RSI' in price_df else None,
            "macd_line": price_df['MACD_line'].iloc[-1] if 'MACD_line' in price_df else None,
            "macd_signal": price_df['MACD_signal'].iloc[-1] if 'MACD_signal' in price_df else None,
            "macd_histogram": price_df['MACD_histogram'].iloc[-1] if 'MACD_histogram' in price_df else None,
            "stochastic_k": price_df['%K'].iloc[-1] if '%K' in price_df else None,
            "stochastic_d": price_df['%D'].iloc[-1] if '%D' in price_df else None
        }
        
        state.current_step = "analyze_technicals"
        state.messages.append({
            "role": "assistant", 
            "content": f"Calculated technical indicators for {state.ticker}"
        })
        
        # Store the DataFrame for use in the next step
        state.price_df = price_df.to_dict(orient='records')
        
    except Exception as e:
        state.error = f"Error calculating technical indicators: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error calculating technical indicators: {str(e)}"
        })

    return state


def analyze_technicals(state: MurphyState) -> MurphyState:
    """Analyze all technical aspects of the price data."""
    # Skip if error
    if state.error:
        return state
    
    try:
        price_df = state.price_df  # Get the DataFrame stored in previous step
        price_df = pd.DataFrame(price_df)
        # Run all technical analyses
        state.trend_analysis = analyze_trend(price_df)
        state.support_resistance = calculate_sup_res(price_df)
        state.momentum_analysis = analyze_momentum(price_df)
        state.volume_analysis = analyze_volume(price_df)
        state.pattern_analysis = identify_patterns(price_df)
        
        # Calculate total technical score
        trend_score = state.trend_analysis.get("trend_score", 0)
        momentum_score = state.momentum_analysis.get("momentum_score", 0)
        
        # Volume impact
        volume_impact = 0
        if state.volume_analysis.get("volume_trend") in ["strongly increasing", "increasing"]:
            volume_impact = 2 if trend_score > 0 else -2
        elif state.volume_analysis.get("volume_trend") in ["strongly decreasing", "decreasing"]:
            volume_impact = -2 if trend_score > 0 else 2
            
        # Chart pattern impact
        pattern_impact = 0
        bullish_patterns = ["double bottom", "inverse head and shoulders", "ascending triangle", "ascending channel"]
        bearish_patterns = ["double top", "head and shoulders", "descending triangle", "descending channel"]
        
        for pattern in state.pattern_analysis.get("patterns", []):
            if pattern in bullish_patterns:
                pattern_impact += 2
            elif pattern in bearish_patterns:
                pattern_impact -= 2
                
        # Calculate final score
        total_score = trend_score + momentum_score + volume_impact + pattern_impact
        
        state.total_score = total_score
        state.max_score = 15  # Approximate maximum possible score
        
        state.current_step = "generate_signal"
        state.messages.append({
            "role": "assistant", 
            "content": f"Completed technical analysis for {state.ticker}"
        })
        
        # Clean up DataFrame to prevent serialization issues
        state.price_df = None
        
    except Exception as e:
        state.error = f"Error analyzing technicals: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error analyzing technicals: {str(e)}"
        })
    
    return state



def generate_signal(state: MurphyState) -> Dict:
    """Generate final technical analysis signal."""
    # Skip if error
    if state.error:
        return state.model_dump()
    
    try:
        # Prepare analysis data for LLM
        analysis_data = {
            "ticker": state.ticker,
            "price": state.technical_data.get("last_price"),
            "trend_analysis": state.trend_analysis,
            "support_resistance": state.support_resistance,
            "momentum_analysis": state.momentum_analysis,
            "volume_analysis": state.volume_analysis,
            "pattern_analysis": state.pattern_analysis,
            "total_technical_score": state.total_score,
            "key_indicators": {
                "RSI": state.technical_data.get("rsi"),
                "MACD": {
                    "line": state.technical_data.get("macd_line"),
                    "signal": state.technical_data.get("macd_signal"),
                    "histogram": state.technical_data.get("macd_histogram")
                },
                "Stochastic": {
                    "%K": state.technical_data.get("stochastic_k"),
                    "%D": state.technical_data.get("stochastic_d")
                },
                "Moving_Averages": {
                    "SMA20": state.technical_data.get("sma_20"),
                    "SMA50": state.technical_data.get("sma_50"),
                    "SMA200": state.technical_data.get("sma_200")
                }
            }
        }
        
        # Generate preliminary signal based on total score
        total_score = state.total_score
        
        if total_score >= 8:
            signal = "bullish"
        elif total_score <= -8:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # LLM prompt for Murphy's reasoning and refinement
        prompt = get_murphy_prompt(
            ticker=state.ticker,
            analysis_data=json.dumps(analysis_data, indent=2),
            signal=signal
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=GEMINI_API_KEY
        )
        
        # Get LLM response and parse it
        chain = llm | TradingSignalParser()
        murphy_output = chain.invoke(prompt)
        
        # Update state
        state.output_signal = murphy_output
        state.current_step = "complete"
        
        # Add final message
        state.messages.append({
            "role": "assistant", 
            "content": f"John Murphy technical analysis complete for {state.ticker}: {murphy_output.signal.upper()} with {murphy_output.confidence:.0%} confidence"
        })
        
    except Exception as e:
        state.error = f"Error generating signal: {str(e)}"
        state.current_step = "error"
        state.messages.append({
            "role": "assistant", 
            "content": f"Error generating signal: {str(e)}"
        })
    return state


def handle_error(state: MurphyState) -> MurphyState:
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
def router(state: MurphyState) -> str:
    if state.error:
        return "error_handler"
    
    return state.current_step


def create_murphy_agent(): 

    workflow = StateGraph(MurphyState)

    workflow.add_node("initialize", initialize_state)
    workflow.add_node("fetch_price_data", fetch_price_data)
    workflow.add_node("calculate_technical_indicators", calculate_technical_indicators)
    workflow.add_node("analyze_technicals", analyze_technicals)
    workflow.add_node("generate_signal", generate_signal)
    workflow.add_node("error_handler", handle_error)


    workflow.add_edge("initialize", "fetch_price_data")
    workflow.add_conditional_edges(
        "fetch_price_data",
        router,
        {
            "error_handler": "error_handler",
            "calculate_technical_indicators": "calculate_technical_indicators"
        }
    )
    workflow.add_conditional_edges(
        "calculate_technical_indicators",
        router,
        {
            "error_handler": "error_handler",
            "analyze_technicals": "analyze_technicals"
        }
    )
    workflow.add_conditional_edges(
        "analyze_technicals",
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


def run_murphy_analysis(ticker: str, end_date: str = None) -> Dict:


    agent = create_murphy_agent()
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    initial_state = {"ticker": ticker, "end_date": end_date}
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    # Return the final state
    return final_state.get("output_signal", None)