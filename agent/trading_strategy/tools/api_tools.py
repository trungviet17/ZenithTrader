from typing import Dict, Any
import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
import requests
import pandas as pd 

load_dotenv()


FMP_API_KEY = os.getenv("FMP_API_KEY")
FN_API_KEY = os.environ.get("FINANCIAL_DATASETS_API_KEY")



def get_financial_metrics(ticker: str,  period: str = 'ttm', limit=20) -> Dict[str, Any]:

    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = (
        f'https://api.financialdatasets.ai/financial-metrics'
        f'?ticker={ticker}'
        f'&period={period}'
        f'&limit={limit}'
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    data = response.json()
    return data


def search_line_items( ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10 ) -> Dict[str, Any]:
    
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()

    return data


def get_history_price(ticker: str, period: str = "1y", interval : str = "1d") -> pd.DataFrame: 

    header = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        header["X-API-KEY"] = api_key

    try : 
        url = (
            f'https://api.financialdatasets.ai/prices/'
            f'?ticker={ticker}'
            f'&interval={interval}'
            f'&period={period}'
        )
        response = requests.get(url, headers=header)

        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        price_data = response.json()

        df = pd.DataFrame(price_data['prices'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)


        return df 

    except Exception as e:
        print(f"Error fetching data: {ticker} - {e}")
        return pd.DataFrame()

