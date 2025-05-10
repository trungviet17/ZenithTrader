import os 
from dotenv import load_dotenv



from modules.market_intelligence.state import MarketIntelligenceState
from server.schema import AssetData
from modules.market_intelligence.helper import get_latest_information, get_llm, PastMarketOutputParser, LatestMarketOutputParser, RetrievalInformationParser
from langgraph.prebuilt import ToolNode 
from modules.market_intelligence.tools import get_tools
from modules.market_intelligence.prompt.past_prompt import create_past_prompt_template
from modules.market_intelligence.prompt.latest_prompt import create_latest_prompt_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from modules.market_intelligence.state import ProcessingState
from modules.market_intelligence.prompt.general_prompt import retrieval_prompt



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")




def past_market_intelligent(state : MarketIntelligenceState) -> MarketIntelligenceState:
    
    if state.information is None:
        raise ValueError("Invalid final output. Please check the previous steps.")
    
    state.messages.append({
        "role": "user",
        "content": f"Analyze the past market intelligence for {state.input.asset_name} ({state.input.asset_symbol})"
    })

    state.curr_stage = ProcessingState.PAST_PROCESSING
    prompt = create_past_prompt_template(state.input)
    formatted_prompt = prompt.format(
        asset_name = state.input.asset_name,
        asset_symbol = state.input.asset_symbol,
        past_market_intelligence = state.information,
    ) 

    llm = get_llm(have_tools= False)
    parser = PastMarketOutputParser()

    chain = llm | parser

    try : 
        result = chain.invoke(formatted_prompt)
        state.past_intelligent = result    


    except Exception as e:
        print(f"Error: {e}")
        result = None


    return state 


def latest_market_intelligent(state : MarketIntelligenceState) -> MarketIntelligenceState:
    
    state.curr_stage = ProcessingState.LATEST_PROCESSING
    state.information = get_latest_information(state.input)

    prompt = create_latest_prompt_template(state.input)
    formatted_prompt = prompt.format(
        asset_name = state.input.asset_name,
        asset_symbol = state.input.asset_symbol,
        latest_market_intelligence = state.information,
    )

    
    # add the latest market intelligence to the state

    llm = get_llm(have_tools=False)
    parser = LatestMarketOutputParser()
    chain = llm | parser

    state.messages.append({
        "role" : "user", 
        "content" : f"Analyze the latest market intelligence for {state.input.asset_name} ({state.input.asset_symbol})"
    })


    try : 
        result = chain.invoke(formatted_prompt)
        state.latest_intelligent = result  

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

    
    market_query = state.latest_intelligent.query 
    queries = []
    if market_query.short_term_query:
        queries.append(market_query.short_term_query)
    if market_query.medium_term_query:
        queries.append(market_query.medium_term_query)
    if market_query.long_term_query:
        queries.append(market_query.long_term_query)



    results = []
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", 
                                                   google_api_key = GEMINI_API_KEY)
    

    # load vectorstore 
    vectorstore = Chroma(
        collection_name = "market_intelligence",
        embedding_function= embedding_model,
        persist_directory= "vector_store",
    )


    for i in queries:
        past_information = vectorstore.similarity_search(query = i, k = 2)
        results.append(past_information[0].page_content)
        results.append(past_information[1].page_content)        

    state.information = results

    llm = get_llm()
    retrieval_parser = RetrievalInformationParser()
    tool_description = ""
    for tool in get_tools():
        tool_description += f"{tool.name}: {tool.description}\n"


    prompt = retrieval_prompt.format(
        current_information = state.latest_intelligent.summaries,
        historical_information = state.information,
        tools_description = tool_description,
    )
    chain = llm | retrieval_parser

    state.messages.append({
        "role": "user",
        "content": prompt
    })

    try : 
        result = chain.invoke(prompt)
        
        state.information += "\n" + result

    except Exception as e:
        print(f"Error: {e}")
        result = None
        state.messages.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })

    return state 



def search_tool(state : MarketIntelligenceState) -> MarketIntelligenceState:

    """
    This function is used to process the search tool.
    """
    tools = get_tools()
    tool_node = ToolNode(tools)
       
    return tool_node 




