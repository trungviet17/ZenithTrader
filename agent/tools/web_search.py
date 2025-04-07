from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API")    

tools = [
    TavilySearchResults(
        max_results=1, 
        tavily_api_key = TAVILY_API_KEY
    ),
]