from enum import Enum 
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union 



class TradeDecision(BaseModel): 

    symbol : str 
    action : str 
    quantity : float 
    price : float 
    reasoning : str 
    agent_name : str
    confidence : float 



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