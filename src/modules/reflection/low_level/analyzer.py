from llm.factory import LLMFactory
from prompts.manager import PromptManager
from memory.manager import MemoryManager
from low_level.indicators import TechnicalIndicatorProcessor
import pandas as pd
from typing import Dict, Any, List
import numpy as np
import xml.etree.ElementTree as ET

class LowLevelAnalyzer:
    """Analyze price movements, Kline charts, and market intelligence."""

    def __init__(self):
        self.llm = LLMFactory.create_provider()
        self.prompt_manager = PromptManager()
        self.memory = MemoryManager()
        self.indicator_processor = TechnicalIndicatorProcessor()
        self.prompt_manager.load_template("llr_analysis")
        self.prompt_manager.load_template("llr_output_format")
        self.prompt_manager.load_template("llr_query")

    async def analyze_price_movements(self, symbol: str, market_data: Dict[str, Any], 
                                   asset_info: Dict[str, str]) -> Dict[str, Any]:
        """Analyze price movements across timeframes with market intelligence."""
        # Define timeframes
        periods = {
            "short_term": "7d",
            "medium_term": "1mo",
            "long_term": "3mo"
        }
        
        # Fetch Kline data
        kline_data = await self.indicator_processor.fetch_kline_data(symbol, periods)
        
        # Compute indicators
        indicator_data = await self.indicator_processor.compute_indicators(kline_data)
        
        # Analyze price changes
        price_analysis = {}
        for timeframe, df in kline_data.items():
            if not df.empty:
                price_change = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
                trend = "Increasing" if price_change > 0 else "Decreasing" if price_change < -0 else "Stable"
                candlestick = "Green" if df["Close"].iloc[-1] > df["Open"].iloc[-1] else "Red"
                price_analysis[timeframe] = {
                    "change_pct": price_change,
                    "trend": trend,
                    "candlestick": candlestick,
                    "latest_close": df["Close"].iloc[-1],
                    "high": df["High"].max(),
                    "low": df["Low"].min()
                }
            else:
                price_analysis[timeframe] = {
                    "change_pct": 0,
                    "trend": "N/A",
                    "candlestick": "N/A",
                    "latest_close": None,
                    "high": None,
                    "low": None
                }
        
        # Prepare market intelligence
        latest_mi = market_data.get("latest_market_intelligence", "No latest news available.")
        past_mi = market_data.get("past_market_intelligence", "No past news available.")
        
        # Prepare data for LLM
        short_term_data = (
            f"Price change: {price_analysis['short_term']['change_pct']:.2f}%, "
            f"Trend: {price_analysis['short_term']['trend']}, "
            f"Candlestick: {price_analysis['short_term']['candlestick']}"
        )
        medium_term_data = (
            f"Price change: {price_analysis['medium_term']['change_pct']:.2f}%, "
            f"Trend: {price_analysis['medium_term']['trend']}, "
            f"Candlestick: {price_analysis['medium_term']['candlestick']}"
        )
        long_term_data = (
            f"Price change: {price_analysis['long_term']['change_pct']:.2f}%, "
            f"Trend: {price_analysis['long_term']['trend']}, "
            f"Candlestick: {price_analysis['long_term']['candlestick']}"
        )
        indicators = (
            f"Short-term: MA5={indicator_data['short_term']['ma5']}, "
            f"BB Upper={indicator_data['short_term']['bb_upper']}, "
            f"BB Lower={indicator_data['short_term']['bb_lower']}, "
            f"Bandwidth={indicator_data['short_term']['bandwidth']:.2f}%\n"
            f"Medium-term: MA5={indicator_data['medium_term']['ma5']}, "
            f"BB Upper={indicator_data['medium_term']['bb_upper']}, "
            f"BB Lower={indicator_data['medium_term']['bb_lower']}, "
            f"Bandwidth={indicator_data['medium_term']['bandwidth']:.2f}%\n"
            f"Long-term: MA5={indicator_data['long_term']['ma5']}, "
            f"BB Upper={indicator_data['long_term']['bb_upper']}, "
            f"BB Lower={indicator_data['long_term']['bb_lower']}, "
            f"Bandwidth={indicator_data['long_term']['bandwidth']:.2f}%"
        )
        
        # Call LLM for reasoning
        analysis_prompt = self.prompt_manager.get_prompt(
            "llr_analysis",
            symbol=symbol,
            exchange=asset_info.get("exchange", "Unknown"),
            sector=asset_info.get("sector", "Unknown"),
            latest_market_intelligence=latest_mi,
            past_market_intelligence=past_mi,
            short_term_data=short_term_data,
            medium_term_data=medium_term_data,
            long_term_data=long_term_data,
            indicators=indicators
        )
        reasoning_text = await self.llm.invoke(analysis_prompt, context={"symbol": symbol})
        
        # Parse reasoning (assuming LLM returns structured text)
        reasoning_lines = reasoning_text.split("\n")
        reasoning = {
            "short_term_reasoning": next((line for line in reasoning_lines if "short-term" in line.lower()), "No short-term reasoning provided."),
            "medium_term_reasoning": next((line for line in reasoning_lines if "medium-term" in line.lower()), "No medium-term reasoning provided."),
            "long_term_reasoning": next((line for line in reasoning_lines if "long-term" in line.lower()), "No long-term reasoning provided.")
        }
        
        # Format output
        output_prompt = self.prompt_manager.get_prompt(
            "llr_output_format",
            symbol=symbol,
            short_term_reasoning=reasoning["short_term_reasoning"],
            medium_term_reasoning=reasoning["medium_term_reasoning"],
            long_term_reasoning=reasoning["long_term_reasoning"]
        )
        xml_output = await self.llm.invoke(output_prompt)
        
        # Generate queries
        query_prompt = self.prompt_manager.get_prompt(
            "llr_query",
            symbol=symbol,
            short_term_reasoning=reasoning["short_term_reasoning"],
            medium_term_reasoning=reasoning["medium_term_reasoning"],
            long_term_reasoning=reasoning["long_term_reasoning"]
        )
        query_text = await self.llm.invoke(query_prompt)
        query_lines = query_text.split(";")
        queries = {
            "short_term_query": query_lines[0].strip() if len(query_lines) > 0 else f"Short-term price movement for {symbol}",
            "medium_term_query": query_lines[1].strip() if len(query_lines) > 1 else f"Medium-term price movement for {symbol}",
            "long_term_query": query_lines[2].strip() if len(query_lines) > 2 else f"Long-term price movement for {symbol}"
        }
        query_embeddings = np.random.rand(len(queries), 256)  # Random 256-dim embeddings
        
        # Store analysis
        analysis_data = {
            "ticker": symbol,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "pattern": price_analysis["short_term"]["trend"],
            "insight": reasoning_text,
            "xml_output": xml_output,
            "queries": queries
        }
        self.memory.store_llr([analysis_data])
        
        return {
            "price_analysis": price_analysis,
            "indicator_summary": indicator_data,
            "reasoning": reasoning,
            "xml_output": xml_output,
            "queries": queries,
            "query_embeddings": query_embeddings
        }