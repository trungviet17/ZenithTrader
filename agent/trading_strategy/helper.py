
import sys 
import os
sys.path.append(os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  

import json
import re
from langchain_core.output_parsers import BaseOutputParser
from agent.trading_strategy.state import TradingSignal


class TradingSignalParser(BaseOutputParser):
    def parse(self, text: str) -> TradingSignal:
        
        try:
            # Handle text wrapped in code blocks
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            print(f"Parsed text: {text}")
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            # Parse JSON
            data = json.loads(text)
            
            # Map market sentiment to trading signals
            signal_mapping = {
                "bullish": "buy",
                "bearish": "sell",
                "neutral": "hold"
            }
            
            signal = signal_mapping.get(data.get("signal", "").lower(), "hold")
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            return TradingSignal(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON output: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse trading signal output: {e}")
        


if __name__ == "__main__":
    # Example usage
    parser = TradingSignalParser()
    example_text = """```json
{
    "signal": "bearish",
    "confidence": "0.8",
    "reasoning": "Alright, let's take a look at Apple, ticker AAPL. Now, I like to keep things simple. I only invest in businesses I understand, and I reckon I understand Apple well enough. People love their gadgets. But that doesn't mean we blindly jump in. We need to see if it makes sense at the current price.    First off, the good. They have a strong ROE of 151.3%. That's quite something. A good sign of a business that knows how to make money with the money it has. Their operating margins are strong too. The Moat analysis suggests a solid competitive advantage with stable ROE and operating margins. And management seems decent, buying back shares and paying dividends. These are shareholder-friendly moves, something I always appreciate.

    However, there are some concerning figures. Their debt-to-equity ratio is a hefty 4.0. I prefer companies that aren't overly leveraged. Debt can be a killer, especially when things get tough. The current ratio is also weak at 0.8, which suggests that they may have liquidity issues. I prefer to invest in companies that are financially secure.

    Now, let's talk about valuation. The intrinsic value analysis, using a discounted cash flow model based on owner earnings, comes out to around $1.93 trillion. The current market cap is $3.27 trillion. That means we're looking at a negative margin of safety of over 41%. I like to buy things on sale, at a significant discount to their intrinsic value, at least 30%. This ain't on sale; it's overpriced.

    Furthermore, the consistency analysis reveals inconsistent earnings growth. I look for steady, predictable growth. The PEG ratio is also high, suggesting the company is overvalued based on its projected earnings growth.

    So, while Apple has a strong brand and seems to have a durable moat, the high debt, weak liquidity, and the fact that the market price is way above my calculated intrinsic value gives me pause. We have to be disciplined. No matter how good a business is, it's not a good investment if the price is too high. Therefore, based on my principles of margin of safety and financial strength, I'm inclined to stay away at this price. It is better to miss a good opportunity than lose money because of a bad investment. I'd say this is a **bearish** signal."
}
```"""

    signal = parser.parse(example_text)
  