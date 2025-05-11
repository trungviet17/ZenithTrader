import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager.state import TradeDecision, RiskManagerOut
from langchain_core.output_parsers import BaseOutputParser
import json 
import re

class RiskManagerParser(BaseOutputParser):

    def parse(self, text: str) -> RiskManagerOut : 
        try : 

            if text.strip().startswith("```") : 
                text = text.replace("```json", "").replace("```", "").strip()

            print(f"Parsed text: {text}")
            text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
            data = json.loads(text)

            
            # Extract and clean price value if it's a string
            price_value = data.get("price", 0.0)
            if isinstance(price_value, str):
                # Remove $ and any other non-numeric characters except decimal point
                price_value = re.sub(r'[^\d.]', '', price_value)
            
            trading_decision = TradeDecision(
                symbol=data.get("symbol", ""),
                action=data.get("action", ""),
                quantity=data.get("quantity", 0),
                price=float(price_value),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.0)),
                agent_name="RiskManager", 
                exchange_name=data.get("exchange_name", "")
            )

            risk_manager_out = RiskManagerOut(
                trade_decision=trading_decision,
                reasoning=data.get("adjustment_reasoning", "")
            )

            return risk_manager_out

        except json.JSONDecodeError as e :
            raise ValueError(f"Failed to parse JSON output: {e}")
        

        except Exception as e :
            raise ValueError(f"Failed to parse risk manager output: {e}")
