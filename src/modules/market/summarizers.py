from llm.factory import LLMFactory
from prompts.manager import PromptManager
import pandas as pd
from typing import Dict, Any, List

class MarketSummarizer:
    """Summarize market intelligence data."""

    def __init__(self):
        self.llm = LLMFactory.create_provider()
        self.prompt_manager = PromptManager()
        self.prompt_manager.load_template("market_summary")

    async def summarize(self, symbol: str, price_data: pd.DataFrame, news_data: List[Dict]) -> Dict[str, str]:
        """Generate summary using LLM."""
        price_summary = f"Latest close: {price_data['Close'].iloc[-1]:.2f}"
        news_summary = "; ".join([item["text"] for item in news_data])
        
        prompt = self.prompt_manager.get_prompt(
            "market_summary",
            symbol=symbol,
            price_data=price_summary,
            news_data=news_summary
        )
        response = await self.llm.invoke(prompt, context={"symbol": symbol})
        return {"price_summary": price_summary, "news_summary": response}