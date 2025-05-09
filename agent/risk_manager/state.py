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
    exchange_name : str = Field(default="")


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
    holding: Optional[Dict[str, int]] = None
    risk_profile: Optional[RiskProfile] = None
    mitigation_plan: Optional[Dict[str, float]] = None
    approval_status: Optional[str] = None
    reasoning: Optional[str] = None
    next_step: str = Field(default="initialize_state")
    message: Optional[List[str]] = []


class RiskManagerOut(BaseModel): 

    trade_decision: TradeDecision
    reasoning: str 

