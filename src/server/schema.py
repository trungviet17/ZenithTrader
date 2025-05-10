from pydantic import BaseModel 



class AssetData(BaseModel):
    asset_symbol: str 
    asset_name : str 
    asset_type : str 
    asset_exchange : str 
    asset_sector : str
    asset_industry : str 
    asset_description : str 
    