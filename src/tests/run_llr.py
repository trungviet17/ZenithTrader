import asyncio
from src.modules.reflection.low_level.agent import LowLevelReflectionAgent
from src.modules.market.fetchers import fetch_market_data

async def main():
    market_data = await fetch_market_data(symbol="AAPL")
    market_data["latest_market_intelligence"] = "Positive earnings report released today."
    market_data["past_market_intelligence"] = "Mixed news over the past month."
    
    asset_info = {
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "description": "Apple Inc. designs and sells consumer electronics."
    }
    
    agent = LowLevelReflectionAgent()
    result = await agent.run(symbol="AAPL", market_data=market_data, asset_info=asset_info)
    print("Low-Level Reflection Output:")
    print("XML Output:", result["xml_output"])
    print("Queries:", result["queries"])

if __name__ == "__main__":
    asyncio.run(main())