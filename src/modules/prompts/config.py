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
            },
            "llr_analysis": {
                "template": (
                    "You are an expert trader analyzing {symbol} listed on {exchange} in the {sector} sector.\n"
                    "Based on the following data:\n"
                    "Latest Market Intelligence: {latest_market_intelligence}\n"
                    "Past Market Intelligence: {past_market_intelligence}\n"
                    "Kline Chart Analysis:\n"
                    "- Short-term ({short_term_data})\n"
                    "- Medium-term ({medium_term_data})\n"
                    "- Long-term ({long_term_data})\n"
                    "Technical Indicators: {indicators}\n"
                    "Analyze step-by-step the reasoning behind price movements for each timeframe. "
                    "Focus on the impact of market intelligence (especially latest news) and Kline chart trends. "
                    "Provide concise reasoning (max 300 tokens per timeframe) for short-term, medium-term, and long-term."
                )
            },
            "llr_output_format": {
                "template": (
                    "Format the following analysis for {symbol} into a valid XML object:\n"
                    "Short-term Reasoning: {short_term_reasoning}\n"
                    "Medium-term Reasoning: {medium_term_reasoning}\n"
                    "Long-term Reasoning: {long_term_reasoning}\n"
                    "Query Sentence: The key sentence should be utilized to retrieve past reasoning for price movements.\n"
                    "Output should follow this XML structure:\n"
                    "<output>\n"
                    "  <map name=\"reasoning\">\n"
                    "    <string name=\"short_term_reasoning\">{short_term_reasoning}</string>\n"
                    "    <string name=\"medium_term_reasoning\">{medium_term_reasoning}</string>\n"
                    "    <string name=\"long_term_reasoning\">{long_term_reasoning}</string>\n"
                    "  </map>\n"
                    "  <string name=\"query\">The key sentence should be utilized to retrieve past reasoning for price movements.</string>\n"
                    "</output>"
                )
            },
            "llr_query": {
                "template": (
                    "Based on the reasoning for {symbol}:\n"
                    "Short-term: {short_term_reasoning}\n"
                    "Medium-term: {medium_term_reasoning}\n"
                    "Long-term: {long_term_reasoning}\n"
                    "Generate three concise query sentences (max 100 tokens each) to retrieve past reasoning for price movements. "
                    "Each query should summarize key information from one timeframe's reasoning."
                )
            }
        }

    def get_template(self, name: str) -> Dict[str, Any]:
        return self.templates.get(name, {})