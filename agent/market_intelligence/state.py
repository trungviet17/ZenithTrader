from typing import Union
from pydantic import BaseModel


class LatestMarketOutput(BaseModel): 

    query : list[str]
    analysis : list[str] 
    summaries: str 


class PastMarketOutput(BaseModel):

    analysis: list[str]
    summaries: str



class MarketIntelligenceState(BaseModel): 

    input: str   # thong tin (news)
    output : Union[LatestMarketOutput, PastMarketOutput] # phan tich va tom tat


    
