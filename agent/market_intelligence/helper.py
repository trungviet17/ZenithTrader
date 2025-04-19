import sys 
import os 
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agent.market_intelligence.tools import MarketSearchingTools
from server.schema import AssetData
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.market_intelligence.tools import get_tools
from typing import Union , Type
from agent.market_intelligence.state import LatestMarketOutput, PastMarketOutput, MarketQuery 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from typing import Dict, Any
from langchain_core.output_parsers import BaseOutputParser
import json 
import re 


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




class PastMarketOutputParser(BaseOutputParser):
    
    def parse(self, text: str) -> LatestMarketOutput:
        """
        Parses the output text into a structured format.
        """
        try:
            # Assuming the output is in JSON format
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)

            raw_analysis = data.get("analysis", "")
            analysis_list = re.findall(r'ID:\s*\d+\s*-\s*([^I]+)', raw_analysis)
            


            return PastMarketOutput(
                analysis=[item.strip() for item in analysis_list if item.strip()],
                summaries=data.get("summary", "")
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}")


class LatestMarketOutputParser(BaseOutputParser):

    def parse(self, text: str) -> LatestMarketOutput:
        """
        Parses the output text into a structured format.
        """
        try:
    
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)

            raw_analysis = data.get("analysis", "")
            analysis_list = re.findall(r'ID:\s*\d+\s*-\s*([^I]+)', raw_analysis)
            query = data.get("query", {})
            return LatestMarketOutput(
                query=MarketQuery(
                    short_term_query=query.get("short_term_query", ""),
                    medium_term_query=query.get("medium_term_query", ""),
                    long_term_query=query.get("long_term_query", "")
                ),
                analysis=[item.strip() for item in analysis_list if item.strip()],
                summaries=data.get("summary", "")
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}")



class RetrievalInformationParser(BaseOutputParser):

    def parse(self, text: str) -> str: 

        try : 
            if text.strip().startswith("```"):
                text = re.sub(r"```json|```", "", text).strip()
            data = json.loads(text)
            if not bool(data.get("information_sufficient")): 
                return ""
            return data.get("adding_information", "")
        

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse output: {e}")



def get_latest_information(data: AssetData) -> str:

    tools = MarketSearchingTools()
    result = "Financial Report\n"
    result += "==================\n"
    idx = 0 
    
    vector_store = Chroma(
        embedding_function = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key = GEMINI_API_KEY),
        persist_directory = "agent/market_intelligence/vector_store",
        collection_name = "market_intelligence"
    )

    # get financial report 
    financial_report = tools.financial_report_tool(data.asset_symbol)   
    vector_store.add_texts(texts = financial_report, metadatas = [{"id": str(uuid4())} for _ in range(len(financial_report))])

    
    result += f"ID :{idx} " + financial_report + "\n\n"
    idx += 1

    # # get sentiment news 
    # sentiments_news = tools.sentiment_analysis_tool(ticker = data.asset_symbol, 
    #                                                 limit = 5 )
    # result  += "Sentiment Analysis\n"
    # result += "==================\n"
    # result += sentiments_news[0] + "\n\n"

    # vector_store.add_texts(texts = sentiments_news[1:], metadatas = [{"id": str(uuid4())} for _ in range(len(sentiments_news[1:]))])


    # for news in sentiments_news[1:]:
    #     result += f"ID :{idx} " + news + "\n\n"
    #     idx += 1

    
    # # get history data 
    # history_data = tools.get_price(data.asset_symbol)
    # result += f"ID :{idx} " + history_data + "\n\n"
    # vector_store.add_texts(texts = [history_data], metadatas = [{"id": str(uuid4())}])

    return result 



def get_llm(model : str = "gemini-2.0-flash", have_tools: bool = True) :

    
    llm = ChatGoogleGenerativeAI(model = model, api_key = GEMINI_API_KEY, temperature = 0.5)
    if have_tools : 
        tools = get_tools() 

        return llm.bind_tools(tools)

    return llm 




