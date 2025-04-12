from typing import Dict, Any

class ToolConfig:
    """Configuration for tools."""

    def __init__(self):
        self.tools = {
            "market_data": {
                "name": "MarketDataTool",
                "description": "Fetch market data including stock prices and news"
            }
        }

    def get_tool_config(self, name: str) -> Dict[str, Any]:
        return self.tools.get(name, {})