import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.risk_manager.state import TradeDecision, RiskManagerOut
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

            trading_decision = TradeDecision(
                symbol=data.get("symbol", ""),
                action=data.get("action", ""),
                quantity=data.get("quantity", 0),
                price=data.get("price", 0.0),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.0)),
                agent_name="RiskManager"
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
