from prompt.main_prompt import TRADING_AGENT_PROMPT_V1
from tools.web_search import tools
from general.node import chatnode
from general.state import GeneralState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


workflow = StateGraph(GeneralState) 


workflow.add_node("chatbot", chatnode)


tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)


workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", tools_condition, {"tools" : "tools", END : END})
workflow.add_edge("chatbot", "tools")


graph = workflow.compile()







