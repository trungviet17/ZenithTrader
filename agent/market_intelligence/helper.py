import sys 
import os 
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agent.market_intelligence.tools import MarketSearchingTools
from server.schema import AssetData
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.market_intelligence.tools import get_tools
from typing import Union , Type
from agent.market_intelligence.state import LatestMarketOutput, PastMarketOutput
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




def get_latest_information(data: AssetData) -> str:

    tools = MarketSearchingTools()
    result = "Financial Report\n"
    result += "==================\n"
    idx = 0 
    
    vector_store = Chroma(
        embedding_function = GoogleGenerativeAIEmbeddings(model = "gemini-2.0-flash", api_key = GEMINI_API_KEY),
        persist_directory = "agent/market_intelligence/vectorstore",
        collection_name = "market_intelligence"
    )

    # get financial report 
    financial_report = tools.financial_report_tool(data.asset_symbol)   
    vector_store.add_texts(texts = financial_report, metadatas = [{"id": str(uuid4())} for _ in range(len(financial_report))])

    for report in financial_report:
    
        result += f"ID :{idx} " + report + "\n\n"
        idx += 1

    # get sentiment news 
    sentiments_news = tools.sentiment_analysis_tool(ticker = data.asset_symbol, 
                                                    limit = 5 )
    news += "Sentiment Analysis\n"
    news += "==================\n"
    news += sentiments_news[0] + "\n\n"

    vector_store.add_texts(texts = sentiments_news[1:], metadatas = [{"id": str(uuid4())} for _ in range(len(sentiments_news[1:]))])


    for news in sentiments_news[1:]:
        result += f"ID :{idx} " + news + "\n\n"
        idx += 1

    
    # get history data 
    history_data = tools.get_price(data.asset_symbol)
    result += f"ID :{idx} " + history_data + "\n\n"
    vector_store.add_texts(texts = [history_data], metadatas = [{"id": str(uuid4())}])

    return result 



def get_llm_with_tools(model : str = "gemini-2.0-flash", output_format: Type[Union[LatestMarketOutput, PastMarketOutput]] = LatestMarketOutput):
    
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key = GEMINI_API_KEY, temperature = 0.5)
    llm = llm.with_structured_output(output_format)
    tools = get_tools() 
    llm_with_tools = llm.bind(tools = tools)
    return llm_with_tools


