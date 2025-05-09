from enum import Enum 
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union 



class TradeDecision(BaseModel): 

    symbol : str 
    action : str = Field(default="")
    quantity : int = Field(default=0)
    price : float = Field(default=0)
    reasoning : str = Field(default="")
    agent_name : str = "RiskManager"
    confidence : float = Field(gt=0, le=1, default=0.5)



class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class RiskFactor(str, Enum):
    MARKET_VOLATILITY = "market_volatility"
    LIQUIDITY = "liquidity"
    COUNTERPARTY = "counterparty"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    REGULATORY = "regulatory"

class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    confidence: float = Field(gt=0, le=1)
    reasoning: str
    mitigation_suggestions: List[str] = []

class RiskProfile(BaseModel):
    overall_risk: RiskLevel
    factor_assessments: Dict[RiskFactor, RiskAssessment]
    summary: str
    recommendations: List[str]



class RiskManagerState(BaseModel): 

    trade_decision: TradeDecision
    risk_profile: Optional[RiskProfile]
    mitigation_plan: Optional[Dict]
    human_input_required: bool
    approval_status: Optional[str]
    reasoning: Optional[str]
    next_step: str
    message: Optional[List[str]] = None


class RiskManagerOut(BaseModel): 

    trade_decision: TradeDecision
    reasoning: str 

