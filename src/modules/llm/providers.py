from langchain_google_genai import ChatGoogleGenerativeAI
from llm.config import LLMConfig
from tool.market_tools import MarketDataTool
from typing import Dict, Any

class LLMProvider:
    """Manage LLM interactions without MCP."""

    def __init__(self, provider: str = "google"):
        self.config = LLMConfig().get_config(provider)
        self.llm = ChatGoogleGenerativeAI(
            model=self.config["model"],
            google_api_key=self.config["api_key"],
            temperature=self.config["temperature"]
        )

    async def invoke(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Invoke LLM with optional context from tools."""
        if context and "symbol" in context:
            market_data = MarketDataTool.get_stock_price(context["symbol"], period="1d")
            prompt = f"{prompt}\nMarket Data: {market_data.to_dict()}"
        return (await self.llm.ainvoke(prompt)).content