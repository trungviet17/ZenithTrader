import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agent.market_intelligence.nodes import past_market_intelligence_retrieval, past_market_intelligent, latest_market_intelligent, search_tool
from langgraph.graph import StateGraph, START, END
from agent.market_intelligence.state import MarketIntelligenceState
from langgraph.prebuilt import ToolNode, tools_condition

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

    graph_builder.add_conditional_edges("latest_market_intelligent", tools_condition, ["search_tool", "past_market_intelligence_retrieval"])
    graph_builder.add_conditional_edges("past_market_intelligent", tools_condition, ["search_tool", END])


    graph = graph_builder.compile()
    return graph 






