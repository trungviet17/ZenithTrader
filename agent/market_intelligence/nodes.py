import sys 
import os 
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from agent.market_intelligence.state import MarketIntelligenceState
from server.schema import AssetData
from agent.market_intelligence.helper import get_latest_information, get_llm_with_tools, PastMarketOutputParser, LatestMarketOutputParser
from langgraph.prebuilt import ToolNode 
from agent.market_intelligence.tools import get_tools, ticket_overview_tool
from agent.market_intelligence.prompt.past_prompt import create_past_prompt_template
from agent.market_intelligence.prompt.latest_prompt import create_latest_prompt_template
from agent.market_intelligence.state import LatestMarketOutput, PastMarketOutput
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from agent.market_intelligence.state import ProcessingState



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




def past_market_intelligent(state : MarketIntelligenceState) -> MarketIntelligenceState:
    
    if state.past_infor is None:
        raise ValueError("Invalid final output. Please check the previous steps.")
    
    state.messages.append({
        "role": "user",
        "content": f"Analyze the past market intelligence for {state.input.asset_name} ({state.input.asset_symbol})"
    })

    state.curr_stage = ProcessingState.PAST_PROCESSING
    prompt = create_past_prompt_template(state.input)
    formatted_prompt = prompt.format(
        asset_name = state.data.asset_name,
        asset_symbol = state.data.asset_symbol,
        past_market_intelligent = state.past_intelligent.past_infor,
    ) 

    llm = get_llm_with_tools()
    parser = PastMarketOutputParser()

    chain = llm | parser

    try : 
        result = chain.invoke(formatted_prompt)
        state.past_intelligent = result    

        state.messages.append({
            "role": "assistant",
            "content": "if {result.summaries} is not enough, you should use tools to search for more information."
        })

    except Exception as e:
        print(f"Error: {e}")
        result = None


    return state 




def latest_market_intelligent(state : MarketIntelligenceState) -> MarketIntelligenceState:
    
    
    state.curr_stage = ProcessingState.LATEST_PROCESSING
    latest_market_intelligent = get_latest_information(state.input)
    if not state.messages:
        state.messages = [{
            "role": "user",
            "content": f"Analyze the latest market intelligence for {state.input.asset_name} ({state.input.asset_symbol})"
        }]


    prompt = create_latest_prompt_template(state.input)
    formatted_prompt = prompt.format(
        asset_name = state.input.asset_name,
        asset_symbol = state.input.asset_symbol,
        latest_market_intelligence = latest_market_intelligent
    )

    
    # add the latest market intelligence to the state

    llm = get_llm_with_tools()
    parser = LatestMarketOutputParser()
    chain = llm | parser

    try : 
        result = chain.invoke(formatted_prompt)
        state.latest_intelligent = result  

        state.messages.append({
            "role": "assistant",
            "content":f"if {result.summaries} is not enough, you should use tools to search for more information.", 

        })


    except Exception as e:
        print(f"Error: {e}")
        result = None
        state.messages.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })

    return state 




def past_market_intelligence_retrieval(state : MarketIntelligenceState) -> MarketIntelligenceState:

    state.curr_stage = ProcessingState.RETRIEVAL_PROCESSING
    if state.latest_intelligent is None or state.latest_intelligent.query is None:
        raise ValueError("Invalid final output. Please check the previous steps.")

    state.messages.append({
        "role": "user",
        "content": f"Retrieve the past market intelligence for {state.input.asset_name} ({state.input.asset_symbol})"
    })
    query = state.latest_intelligent.query 
    results = []
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", 
                                                   google_api_key = GEMINI_API_KEY)
    

    # load vectorstore 
    vectorstore = Chroma(
        collection_name = "market_intelligence",
        embedding_function= embedding_model,
        persist_directory= "agent/market_intelligence/vector_store",
    )


    for i in query:
        past_information = vectorstore.similarity_search(query = i, k = 2)
        results.append(past_information)        

    state.past_infor = results

    state.messages.append({
        "role": "system",
        "content": f"Retrieved {len(results)} historical data points for analysis"
    })


    return state 


    


def search_tool(state : MarketIntelligenceState) -> MarketIntelligenceState:

    """
    This function is used to process the search tool.
    """
    tools = [ticket_overview_tool]
    tool_node = ToolNode(tools)
       
    return tool_node 




