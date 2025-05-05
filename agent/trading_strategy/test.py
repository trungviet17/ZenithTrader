from typing import Dict
from dotenv import load_dotenv
import os 
import requests


load_dotenv()




if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading_strategy folder
    api_dir = os.path.join(os.path.dirname(current_dir), "api")  # api folder
    
    metrics_file = os.path.join(api_dir, "_financial_metrics.json")

    print(metrics_file)