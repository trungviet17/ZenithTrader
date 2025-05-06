from pydantic import BaseModel, Field
from typing import Literal, Optional, List,  Dict


class TradingSignal(BaseModel): 
    signal : Literal['buy', 'sell', 'hold']
    confidence : float 
    reasoning : str 


class BuffettState(BaseModel): 
    # input 
    ticker: str
    end_date: Optional[str] = None

    # fetch data 
    metrics: Optional[List[Dict]] = None
    financial_line_items: Optional[List[Dict]] = None
    market_cap: Optional[float] = None

    # analysis 
    fundamental_analysis: Optional[Dict] = None
    consistency_analysis: Optional[Dict] = None
    moat_analysis: Optional[Dict] = None
    management_analysis: Optional[Dict] = None
    intrinsic_value_analysis: Optional[Dict] = None
    margin_of_safety: Optional[float] = None
    total_score: Optional[float] = None
    max_score: Optional[float] = None

    # output
    error: Optional[str] = None
    output_signal: Optional[TradingSignal] = None

    # tracking 
    current_step: str = "initialize"
    messages: List[Dict] = Field(default_factory=list)




class MurphyState(BaseModel):
    # input 
    ticker: str
    end_date: Optional[str] = None
    interval: Optional[str] = "day"  # "1d", "1h", etc.

    # fetch data 
    price_history: Optional[Dict] = None  
    technical_data: Optional[Dict] = None 

    # analysis 
    price_df : Optional[List[Dict]] = None
    trend_analysis: Optional[Dict] = None
    support_resistance: Optional[Dict] = None
    momentum_analysis: Optional[Dict] = None
    volume_analysis: Optional[Dict] = None
    pattern_analysis: Optional[Dict] = None
    total_score: Optional[float] = None
    max_score: Optional[float] = None

    # output
    error: Optional[str] = None
    output_signal: Optional[TradingSignal] = None

    # tracking 
    current_step: str = "initialize"
    messages: List[Dict] = Field(default_factory=list)







