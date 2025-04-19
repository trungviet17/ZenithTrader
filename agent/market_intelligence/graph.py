import sys, os 
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agent.market_intelligence.nodes import past_market_intelligence_retrieval, past_market_intelligent, latest_market_intelligent, search_tool
from langgraph.graph import StateGraph, START, END
from agent.market_intelligence.state import MarketIntelligenceState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agent.market_intelligence.tools import MarketSearchingTools
from server.schema import AssetData
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



def create_graph(): 
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





