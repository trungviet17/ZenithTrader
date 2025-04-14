from langchain_core.tools import tool
import pandas as pd
import numpy as np

@tool
def calculate_technical_indicators(data: dict, indicator: str, **kwargs) -> dict:
    """Calculate technical indicators (MA, BB)."""
    df = pd.DataFrame(data)
    
    if indicator == "ma":
        period = kwargs.get("period", 20)
        df[f"MA_{period}"] = df["Close"].rolling(window=period).mean()
    elif indicator == "bb":
        period = kwargs.get("period", 20)
        df["MA_20"] = df["Close"].rolling(window=period).mean()
        df["StdDev"] = df["Close"].rolling(window=period).std()
        df["BB_Upper"] = df["MA_20"] + 2 * df["StdDev"]
        df["BB_Lower"] = df["MA_20"] - 2 * df["StdDev"]
    
    return df.to_dict()