from tool.market_tools import MarketDataTool
from tool.calculator import calculate_technical_indicators
from typing import Dict, Any
import pandas as pd

class TechnicalIndicatorProcessor:
    """Process Kline data and compute technical indicators."""

    async def fetch_kline_data(self, symbol: str, periods: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Fetch Kline data for multiple timeframes."""
        data = {}
        for timeframe, period in periods.items():
            market_data = MarketDataTool.get_stock_price(symbol, period)
            data[timeframe] = market_data
        return data

    async def compute_indicators(self, kline_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Compute MA and Bollinger Bands for each timeframe."""
        results = {}
        for timeframe, df in kline_data.items():
            if df.empty:
                results[timeframe] = {"ma5": "N/A", "bb_upper": "N/A", "bb_lower": "N/A", "bandwidth": "N/A"}
                continue
            
            # Calculate Moving Average (MA5)
            ma_data = calculate_technical_indicators.invoke({
                "data": df.to_dict(),
                "indicator": "ma",
                "period": 5
            })
            df = pd.DataFrame(ma_data)
            
            # Calculate Bollinger Bands
            bb_data = calculate_technical_indicators.invoke({
                "data": df.to_dict(),
                "indicator": "bb",
                "period": 20
            })
            df = pd.DataFrame(bb_data)
            
            # Calculate bandwidth
            bb_upper = df["BB_Upper"].iloc[-1] if "BB_Upper" in df else None
            bb_lower = df["BB_Lower"].iloc[-1] if "BB_Lower" in df else None
            bandwidth = (bb_upper - bb_lower) / df["MA_20"].iloc[-1] * 100 if bb_upper and bb_lower and "MA_20" in df else "N/A"
            
            results[timeframe] = {
                "ma5": df["MA_5"].iloc[-1] if "MA_5" in df and not pd.isna(df["MA_5"].iloc[-1]) else "N/A",
                "bb_upper": bb_upper if bb_upper else "N/A",
                "bb_lower": bb_lower if bb_lower else "N/A",
                "bandwidth": bandwidth
            }
        return results