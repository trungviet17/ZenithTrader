import sys 
import os 
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from typing import Union, Optional 
from pydantic import BaseModel
from enum import Enum 
from server.schema import AssetData
from pydantic import Field

class ProcessingState(str, Enum): 
    INIT_STATE = "init_state"
    PAST_PROCESSING = "past_processing"
    LATEST_PROCESSING = "latest_processing"
    RETRIEVAL_PROCESSING = "retrieval_processing"



class MarketQuery(BaseModel):
    """
    MarketQuery represents a structured query for market intelligence analysis.
    """
    short_term_query: Optional[str] = Field(None, description="Query for short-term market analysis")
    medium_term_query: Optional[str] = Field(None, description="Query for medium-term market analysis")
    long_term_query: Optional[str] = Field(None, description="Query for long-term market analysis")








class LatestMarketOutput(BaseModel): 
    """
    LatestMarketOutput represents the structured output of the latest market intelligence analysis.
    """
    query: MarketQuery = Field(..., description="List of queries or questions generated for the latest market data")
    analysis: list[str] = Field(..., description="Analytical insights or findings related to the latest market data")
    summaries: str = Field(..., description="Concise summary of the latest market intelligence output")

class PastMarketOutput(BaseModel):
    """
    PastMarketOutput represents the structured output of the past market intelligence analysis.
    """
    analysis: list[str] = Field(..., description="Analytical insights or findings related to historical market data")
    summaries: str = Field(..., description="Concise summary of the past market intelligence output")



class MarketIntelligenceState(BaseModel): 

    input: AssetData 
    
    curr_stage : ProcessingState = ProcessingState.INIT_STATE

    past_intelligent : Optional[PastMarketOutput] = None
    latest_intelligent : Optional[LatestMarketOutput] = None

    past_infor : Optional[str] = None
    messages: List[Dict[str, Any]] = []

