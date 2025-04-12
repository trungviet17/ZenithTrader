from typing import Dict, Any

class PromptConfig:
    """Configuration for prompt templates."""

    def __init__(self):
        self.templates = {
            "market_summary": {
                "template": (
                    "Summarize the following market data for {symbol}:\n"
                    "Price Data: {price_data}\n"
                    "News Data: {news_data}\n"
                    "Provide a concise summary including price trends and news sentiment."
                )
            },
            "diversified_query": {
                "template": (
                    "Generate diversified queries based on this summary:\n"
                    "{summary}\n"
                    "Include short-term impact, trend analysis, and historical events."
                )
            }
        }

    def get_template(self, name: str) -> Dict[str, Any]:
        return self.templates.get(name, {})