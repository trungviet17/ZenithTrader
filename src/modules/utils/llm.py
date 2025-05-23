from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



class LLM: 
    
    @staticmethod 
    def get_gemini_llm(model_name : str = "gemini-2.0-flash", temperature: float = 0.7, model_index: int = 1):
        key = os.getenv("GEMINI_API_KEY_1")

        if model_index == 1: 
            key = os.getenv("GEMINI_API_KEY_1")
        elif model_index == 2:
            key = os.getenv("GEMINI_API_KEY_2")
        elif model_index == 3:
            key = os.getenv("GEMINI_API_KEY_3")


        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key = key ,
            temperature=temperature,
        ) 


    @staticmethod
    def get_llama_llm(model_name: str =   "llama-3.1-8b-instant", temperature: float = 0.7):

        return ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature= temperature,
        )

    @staticmethod
    def get_deepseek_llm(model_name: str = "deepseek-3.1-8b-instant", temperature: float = 0.7):

        return ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature= temperature,
        )

    @staticmethod
    def get_gemini_embedding(model_name: str = "models/text-embedding-004"):
        key = os.getenv("GEMINI_API_KEY_1")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key= key, 

        )  
    





    





