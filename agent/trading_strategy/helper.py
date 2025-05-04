
import json
import re
from langchain_core.output_parsers import BaseOutputParser
from agent.trading_strategy.state import TradingSignal

class TradingSignalParser(BaseOutputParser):
    def parse(self, text: str) -> TradingSignal:
        """
        Parses JSON output containing trading signals into a TradingSignal object.
        
        Expected input format:
        {
            "signal": "bullish" | "bearish" | "neutral",
            "confidence": float between 0 and 1,
            "reasoning": "detailed explanation"
        }
        """
        try:
            # Handle text wrapped in code blocks
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            
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