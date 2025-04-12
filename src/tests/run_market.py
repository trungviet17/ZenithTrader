import asyncio
from src.modules.market.agent import MarketAgent

async def main():
    agent = MarketAgent()
    result = await agent.run(symbol="AAPL")
    print("Market Intelligence Result:", result)

if __name__ == "__main__":
    asyncio.run(main())