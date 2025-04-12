import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any

class MarketDataTool:
    """Tool to fetch market data from Yahoo Finance."""

    @staticmethod
    def get_stock_price(symbol: str, period: str = "1d") -> pd.DataFrame:
        """Fetch stock price data from Yahoo Finance."""
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data

    @staticmethod
    def get_market_data(symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Fetch price and simulated news data."""
        price_data = MarketDataTool.get_stock_price(symbol, period)
        news_data = [
            {"ticker": symbol, "date": datetime.now().strftime("%Y-%m-%d"), "text": f"Sample news about {symbol}"}
        ]
        return {"price_data": price_data, "news_data": news_data}