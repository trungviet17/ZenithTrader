from modules.market_intelligence.nodes import past_market_intelligence_retrieval, past_market_intelligent, latest_market_intelligent, search_tool
from langgraph.graph import StateGraph, START, END
from modules.market_intelligence.state import MarketIntelligenceState
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



def run_market_intelligence_agent(symbol: str) -> MarketIntelligenceState:
    
    graph = create_market_intelligence_agent()

    try: 
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




if __name__ == "__main__":

    symbol = "AAPL"
    past_market_analysis, latest_market_analysis = run_market_intelligence_agent(symbol)
    print("Past Market Analysis: ", past_market_analysis)
    print("Latest Market Analysis: ", latest_market_analysis)






