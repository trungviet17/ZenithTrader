import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from typing import Union, Optional 
from pydantic import BaseModel
from enum import Enum 
from server.schema import AssetData

class ProcessingState(str, Enum): 
    INIT_STATE = "init_state"
    PAST_PROCESSING = "past_processing"
    LATEST_PROCESSING = "latest_processing"
    RETRIEVAL_PROCESSING = "retrieval_processing"



class LatestMarketOutput(BaseModel): 

    query : list[str]
    analysis : list[str] 
    summaries: str 


class PastMarketOutput(BaseModel):

    analysis: list[str]
    summaries: str



class MarketIntelligenceState(BaseModel): 

    input: AssetData 
    
    curr_stage : ProcessingState = ProcessingState.INIT_STATE

    past_intelligent : Optional[PastMarketOutput] = None
    latest_intelligent : Optional[LatestMarketOutput] = None

    past_infor : Optional[str] = None

