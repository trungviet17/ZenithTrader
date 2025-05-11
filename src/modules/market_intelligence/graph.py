from modules.market_intelligence.nodes import past_market_intelligence_retrieval, past_market_intelligent, latest_market_intelligent, search_tool
from langgraph.graph import StateGraph, START, END
from modules.market_intelligence.state import LatestMarketOutput, MarketIntelligenceState, MarketQuery, PastMarketOutput, ProcessingState
from modules.market_intelligence.tools import MarketSearchingTools
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from modules.market_intelligence.tools import MarketSearchingTools
from server.schema import AssetData
import os
from dotenv import load_dotenv


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




def create_market_intelligence_agent(): 
    graph_builder = StateGraph(MarketIntelligenceState)

    # node 
    graph_builder.add_node("latest_market_intelligent", latest_market_intelligent)
    graph_builder.add_node("past_market_intelligent", past_market_intelligent)
    graph_builder.add_node("past_market_intelligence_retrieval", past_market_intelligence_retrieval)
    graph_builder.add_node("search_tool", search_tool)

    # edges 
    graph_builder.add_edge(START, "latest_market_intelligent")
    graph_builder.add_edge("latest_market_intelligent", "past_market_intelligence_retrieval")
    graph_builder.add_edge("past_market_intelligence_retrieval", "past_market_intelligent")


    graph_builder.add_edge("search_tool", "past_market_intelligence_retrieval")

    
    graph_builder.add_conditional_edges("past_market_intelligence_retrieval", tools_condition, {
        "tools" : "search_tool",
        END: "past_market_intelligent"
    })
    graph_builder.add_edge("past_market_intelligent", END)
    graph = graph_builder.compile()
    return graph 


# sample data for testing 
def create_sample_vectorstore(data: AssetData): 
    vector_store = Chroma(
        embedding_function = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key = GEMINI_API_KEY),
        persist_directory = "agent/market_intelligence/vector_store",
        collection_name = "market_intelligence"
    )


    tools = MarketSearchingTools()

    sample_data = tools.news_api_tool(data, hours_ago= 72)


    vector_store.add_texts(texts = sample_data)

    return vector_store



def run_market_intelligence_agent(symbol: str, data: AssetData, testing = True) -> MarketIntelligenceState:
    
    if testing: 
        result = MarketIntelligenceState(
            input=AssetData(
                asset_symbol='AAPL',
                asset_name='Apple Inc',
                asset_type='Common Stock',
                asset_exchange='NASDAQ',
                asset_sector='TECHNOLOGY',
                asset_industry='ELECTRONIC COMPUTERS',
                asset_description="Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totalling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook."
            ),
            curr_stage=ProcessingState.PAST_PROCESSING,
            past_intelligent=PastMarketOutput(
                analysis=[],
                summaries='There is no market intelligence provided to analyze and summarize. Therefore, I cannot provide any analysis or summary. I need market intelligence data to complete the task.'
            ),
            latest_intelligent=LatestMarketOutput(
                query=MarketQuery(
                    short_term_query='N/A',
                    medium_term_query='N/A',
                    long_term_query='Fiscal year ending 2024-09-30: current ratio, debt ratio, debt/EBITDA, revenue, net income, R&D spending. Quarterly snapshot ending 2025-03-31: revenue, net income, R&D.'
                ),
                analysis=[
                    'Fiscal year ending 2024-09-30 shows a current ratio of 0.87, debt ratio of 0.84, and debt/EB',
                    'Fiscal year ending 2024-09-30 reveals revenue of $391.04B, net income of $93.74B, and R&D spending of $31.37B. LONG-TERM, POS',
                    'Quarterly snapshot ending 2025-03-31 shows revenue of $95.36B, net income of $24.78B, and R&D of $8.55B. LONG-TERM, NEUTRAL.'
                ],
                summaries='The financial report (ID: 000000) presents a mixed picture. While revenue, net income, and R&D spending are strong, indicating a healthy business, the current and debt ratios suggest potential liquidity and leverage concerns. The quarterly snapshot provides a more recent view, but without historical context, it is difficult to assess the trend. Overall sentiment is slightly positive due to strong revenue and profit, but the financial leverage needs to be monitored. LONG-TERM impact.'
            ),
            information=['$', '$', '$', '$', ' ', ' '],
            messages=[
                {'role': 'user', 'content': 'Analyze the latest market intelligence for Apple Inc (AAPL)'},
                {'role': 'user', 'content': '\n    You are an expert financial analyst with deep knowledge of market dynamics. Your task is to evaluate whether the CURRENT and HISTORICAL information provided is sufficient for comprehensive market analysis.\n\n    ## CURRENT INFORMATION\n    The financial report (ID: 000000) presents a mixed picture. While revenue, net income, and R&D spending are strong, indicating a healthy business, the current and debt ratios suggest potential liquidity and leverage concerns. The quarterly snapshot provides a more recent view, but without historical context, it is difficult to assess the trend. Overall sentiment is slightly positive due to strong revenue and profit, but the financial leverage needs to be monitored. LONG-TERM impact.\n\n    ## HISTORICAL INFORMATION (queried based on current context)\n    [\'$\', \'$\', \'$\', \'$\', \' \', \' \']\n\n    ## EVALUATION PROCESS\n    1. ASSESS INFORMATION COMPLETENESS\n    - Evaluate if the combination of current and historical information provides a complete picture of the market conditions.\n    - Check if you have sufficient data on:\n        * Recent market news and events\n        * Financial performance metrics\n        * Market sentiment indicators\n        * Industry-specific trends\n        * Competitive landscape\n        * Macroeconomic factors\n\n    2. IDENTIFY INFORMATION GAPS\n    - Note any critical missing information that would significantly enhance your analysis.\n    - Consider both breadth (range of topics) and depth (detailed insights).\n\n    3. DETERMINE IF ADDITIONAL RESEARCH IS NEEDED\n    - If the current and historical information is still insufficient, decide if using research tools to find more historical data similar to the current context would substantially improve analysis quality.\n\n    ## AVAILABLE TOOLS \n    If you need additional information, you can use these tools to search for more HISTORICAL data relevant to the current situation:\n        web_search: Searches the web for information about markets, companies, or financial topics.\nnews_search: Retrieves recent news articles about a specific asset or company.\nfinancial_report: Retrieves comprehensive financial reports including income statements and balance sheets for a given ticker symbol.\nsentiment_analysis: Analyzes market sentiment for a specific ticker based on recent news and social media.\nticker_overview: Gets comprehensive overview information about a company including its sector, industry, and business description.\nget_price: Retrieves historical price data for a given stock ticker.\n\n\n    ## OUTPUT\n    Provide your evaluation as JSON:\n    ```json\n    {\n        "information_sufficient": "Returns True or False",\n        "adding_information": ["List any additional information needed if the provided information is insufficient"]\n    }\n    ```\n'},
                {'role': 'assistant', 'content': 'Error: can only concatenate str (not "list") to str'},
                {'role': 'user', 'content': 'Analyze the past market intelligence for Apple Inc (AAPL)'}
            ]
        )
        return  result.past_intelligent, result.latest_intelligent
    else : 

        graph = create_market_intelligence_agent()

        try:
            if data is  None: 
                data = MarketSearchingTools().ticket_overview_tool(symbol)

            state = MarketIntelligenceState(
                input = data, 
                past_intelligent= None,
                latest_intelligent = None,
                information= None,
                messages= []
            )

    
        except Exception as e:
            print(f"Error: in using tools {e}")

        result = graph.invoke(state)




    past_market_analysis = result.get("past_intelligent", None)
    latest_market_analysis = result.get("latest_intelligent", None)

    return past_market_analysis, latest_market_analysis




# if __name__ == "__main__":

#     symbol = "AAPL"

#     asset_data = AssetData(
#         asset_symbol='AAPL',
#         asset_name='Apple Inc',
#         asset_type='Common Stock',
#         asset_exchange='NASDAQ',
#         asset_sector='TECHNOLOGY',
#         asset_industry='ELECTRONIC COMPUTERS',
#         asset_description="Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totalling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook."
#     )


#     past_market_analysis, latest_market_analysis = run_market_intelligence_agent(symbol, )
#     print("Past Market Analysis: ", past_market_analysis)
#     print("Latest Market Analysis: ", latest_market_analysis)


graph = create_market_intelligence_agent()






