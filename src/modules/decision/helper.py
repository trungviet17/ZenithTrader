from langchain_core.output_parsers import BaseOutputParser
import json 
import re 
from modules.risk_manager.state import TradeDecision


class DecisionOutputParser(BaseOutputParser):

    def parse (self, text: str) -> dict:

        try: 

            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)
            
            return TradeDecision(
                symbol=data.get("symbol", ""),
                action=data.get("action", ""),
                quantity=data.get("quantity", 0),
                price=data.get("price", 0.0),
                reasoning=data.get("reasoning", ""),
                agent_name=data.get("agent_name", "RiskManager"),
                confidence=data.get("confidence", 0.5),
                exchange_name=data.get("exchange_name", "")
            )




        except json.JSONDecodeError as e:

            raise ValueError(f"Failed to parse output: {e}")

        except Exception as e:

            raise ValueError(f"An unexpected error occurred: {e}")