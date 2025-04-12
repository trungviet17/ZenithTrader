from tool.market_tools import MarketDataTool
from typing import Dict, Any

async def fetch_market_data(symbol: str, period: str = "1d") -> Dict[str, Any]:
    """Fetch market data using MarketDataTool."""
    return MarketDataTool.get_market_data(symbol, period)