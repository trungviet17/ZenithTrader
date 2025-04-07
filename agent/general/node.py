from general.state import GeneralState
from prompt.main_prompt import TRADING_AGENT_PROMPT_V1
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



def _get_model(model_name : str = None):

    if "gemini" in model_name: 
        model = ChatGoogleGenerativeAI(
            temperature=0.0,
            model_name=model_name,
            api_key=GEMINI_API_KEY,
        )

    else : 
        model = ChatGoogleGenerativeAI(
            temperature=0.0,
            model_name="gemini-2.5-pro-preview-03-25",
            api_key=GEMINI_API_KEY,
        )


    return model



def chatnode(state: GeneralState, llm=None):

    if llm is None:
        llm = _get_model()

    history_messages = TRADING_AGENT_PROMPT_V1 + state['messages']
    return {"messages": [llm.invoke(history_messages)]}













