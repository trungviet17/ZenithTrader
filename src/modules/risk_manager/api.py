import yfinance as yf 
import  numpy as np 
from typing import Dict, Any
import json 
import sys, os 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    


def get_volatility_data(ticker: str, period: str = '3mo', interval: str = '1d') -> Dict[str, Any]:

    try:
        # Tải dữ liệu lịch sử của tài sản
        asset = yf.Ticker(ticker)
        hist = asset.history(period=period)
        
        prices = hist["Close"].dropna().tolist()
        volumes = hist["Volume"].dropna().tolist()
        
        # Tải dữ liệu VIX (Chỉ số biến động thị trường)
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1d")
        market_vix = vix_hist["Close"].iloc[-1] if not vix_hist.empty else None
        asset_beta = asset.info.get("beta", None)

        try:
            options_dates = asset.options
            if options_dates:
                opt_chain = asset.option_chain(options_dates[0])  
                calls = opt_chain.calls
                if not calls.empty and "impliedVolatility" in calls.columns:
                    implied_volatility = calls["impliedVolatility"].mean()
                else:
                    implied_volatility = None
            else:
                implied_volatility = None
        except Exception:
            implied_volatility = None

        return {
            "prices": prices,
            "volumes": volumes,
            "market_vix": market_vix,
            "asset_beta": asset_beta,
            "implied_volatility": implied_volatility
        }

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    



def get_liquidity_data(ticker: str, period: str = '3mo', interval: str = '1d') -> Dict[str, Any]:

    try:
        # Tải dữ liệu lịch sử của tài sản
        asset = yf.Ticker(ticker)
        hist = asset.history(period=period)
        
        avg_daily_volume = hist["Volume"].mean() if not hist.empty else None

        info = asset.info
        bid = info.get("bid", None)
        ask = info.get("ask", None)

        if bid != 0 and ask != 0: 
            bid_ask_spread = round((ask - bid) / ask * 100, 4)
        
        else : 
            bid_ask_spread = None


        if bid_ask_spread is not None and avg_daily_volume is not None:
            market_impact_estimate =  (bid_ask_spread / 100) / avg_daily_volume * 1e6
        else:
            market_impact_estimate = None
        


        return {
            "avg_daily_volume": avg_daily_volume,
            "bid_ask_spread": bid_ask_spread,
            "market_impact_estimate": market_impact_estimate
        }
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    

def get_counterparty_data(exchange: str) -> Dict[str, Any]:

    with open("risk_manager/counter_party.json") as f:
        data = json.load(f)

    
    return data.get(exchange, None)


def get_concentration_data(holdings: Dict[str, int]) -> Dict[str, Any]:
    """
    input : 
    """ 
    symbols = list(holdings.keys())
    prices = {}
    sectors = {}
    values = {}
    data = {}


    for symbol in symbols:
        try:
            asset = yf.Ticker(symbol)
            info = asset.info

            current_price = info.get("currentPrice", None)
            sector = info.get("sector", "Unknown")

            print(f"Fetching data for {symbol}: {current_price}, {sector}")

            quantity = holdings[symbol]
            value = quantity * current_price
            
            print(f"Value for {symbol}: {value}")

            prices[symbol] = current_price
            sectors[symbol] = sector
            values[symbol] = value
            
            data.setdefault("holdings", {})[symbol] = {
                "price": current_price,
                "quantity": quantity,
                "value": value,
                "sector": sector
            }

        except KeyError as e:
            print(f"KeyError: {e} for symbol {symbols[symbol]}")
            return None

    total_value = sum(values.values())
    data['total_value'] = total_value

    sector_allocation = {}

    for symbol, value in values.items():
        sector = sectors[symbol]
        sector_allocation[symbol] = sector_allocation.get(sector, 0) + value


    for synmbol in sector_allocation:
        sector_allocation[symbol] = round(sector_allocation[symbol] / total_value * 100, 4)


    data["sector_allocation"] = sector_allocation

    df = yf.download(symbols, period="6mo", interval="1d")
    print(df.head())
    df = df["Close"].dropna()
    corr = df.pct_change().corr()
    
    correlations = {}
    for symbol in symbols:
        others = [s for s in symbols if s != symbol]
        correlations[symbol] = round(corr[symbol][others].mean(), 3) if others else 0.0
    
    data["correlations"] = correlations

    return data




if __name__ == "__main__":


    ticker = "AAPL"
    period = "3mo"

    res = get_liquidity_data(ticker, period)

    print(res)



    







