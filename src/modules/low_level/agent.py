import os
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from uuid import uuid4
import json
from pydantic import BaseModel, Field
from twelvedata import TDClient

from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from modules.utils.llm import LLM 

from dotenv import load_dotenv
load_dotenv()

# --- Khởi tạo LLM, Embeddings và Vector Store ---

llm = LLM.get_gemini_llm(model_index= 2)

embeddings = LLM.get_gemini_embedding()
# --- Thiết lập API Key ---
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./low_level/qdrant_data"
COLLECTION_NAME = "technical_analysis_history"
client = QdrantClient(path=QDRANT_PATH)

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

embedding_size = len(embeddings.embed_query("test"))
create_qdrant_collection(client, COLLECTION_NAME, embedding_size)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6})

# --- Định nghĩa State cho Agent ---
class ReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_context: Optional[str]
    critique: Optional[str]
    response: Optional[str]
    query: Optional[str]
    reflection_data: Optional[str]
    references: Optional[List[str]]
    reflection_iteration: Optional[int]
    max_reflections: int
    final_output: Optional[str]

# --- Tools Phân tích Kỹ thuật ---
@tool
def technical_indicator_analysis(
    symbol: str,
    interval: str = "1day",
    outputsize: int = 30,
    start_date: str = "2025-01-01",
    end_date: str = "2025-03-01"
) -> Dict[str, str]:
    """
    Analyzes patterns based on technical indicators of a stock.
    
    Args:
        symbol (str): Stock symbol to analyze (e.g., "AAPL").
        interval (str): Time interval for data (default is 1 day) (e.g., "8h", "1day", "1week", "1month").
        outputsize (int): Amount of data to retrieve (default is 30).
        start_date (str): Start date for data retrieval ('YYYY-MM-DD' format).
        end_date (str): End date for data retrieval ('YYYY-MM-DD' format).

    Returns:
        dict: Technical analysis results as a dictionary.
    """
    td = TDClient(apikey=TWELVEDATA_API_KEY)

    try:
        ts = (
            td.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                timezone="Exchange",
                dp=5,
                start_date=start_date,
                end_date=end_date
            )
            .with_ma(time_period=9, ma_type="sma", series_type="close")
            .with_ema(time_period=9, series_type="close")
            .with_rsi(time_period=14, series_type="close")
            .with_macd(fast_period=12, slow_period=26, signal_period=9, series_type="close")
            .with_bbands(time_period=20, sd=2, ma_type="sma", series_type="close")
            .with_atr(time_period=14)
            .without_ohlc()
        )

        df = ts.as_pandas()
        analysis = {}

        # MA trend
        if "ma" in df.columns and len(df["ma"].dropna()) >= 5:
            ma_recent = df["ma"].dropna().iloc[-1]
            ma_prev = df["ma"].dropna().iloc[-5]
            trend = "increasing" if ma_recent > ma_prev else "decreasing"
            analysis["MA"] = f"MA has a {trend} trend from {ma_prev:.2f} to {ma_recent:.2f}."

        # EMA trend
        if "ema" in df.columns and len(df["ema"].dropna()) >= 5:
            ema_recent = df["ema"].dropna().iloc[-1]
            ema_prev = df["ema"].dropna().iloc[-5]
            trend = "increasing" if ema_recent > ema_prev else "decreasing"
            analysis["EMA"] = f"EMA has a {trend} trend from {ema_prev:.2f} to {ema_recent:.2f}."

        # RSI
        if "rsi" in df.columns:
            rsi = df["rsi"].dropna().iloc[-1]
            if rsi > 70:
                status = "overbought (may adjust downward)"
            elif rsi < 30:
                status = "oversold (may recover)"
            else:
                status = "neutral"
            analysis["RSI"] = f"Current RSI is {rsi:.2f}, status is {status}."

        # MACD
        if {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
            last = df.dropna(subset=["macd", "macd_signal", "macd_hist"]).iloc[-1]
            signal = "bullish" if last["macd"] > last["macd_signal"] else "bearish"
            analysis["MACD"] = (
                f"MACD: {last['macd']:.2f}, signal: {last['macd_signal']:.2f}, "
                f"histogram: {last['macd_hist']:.2f} → {signal} signal."
            )

        # Bollinger Bands
        if {"upper_band", "middle_band", "lower_band"}.issubset(df.columns):
            bb = df.dropna(subset=["upper_band", "middle_band", "lower_band"]).iloc[-1]
            spread = float(bb["upper_band"]) - float(bb["lower_band"])
            analysis["BollingerBands"] = (
                f"Bollinger Bands have a range of {spread:.2f} points (Upper: {bb['upper_band']}, Lower: {bb['lower_band']})."
            )

        # ATR
        if "atr" in df.columns:
            atr = df["atr"].dropna().iloc[-1]
            analysis["ATR"] = f"ATR indicator is {atr:.2f}, indicating recent average volatility."

        return analysis

    except Exception as e:
        return {"error": str(e)}

    
    
# --- Định nghĩa các công cụ --- 
tools = [technical_indicator_analysis]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

class AnswerQuestionResponder(BaseModel):
    """Schema for the answer"""
    analysis: str = Field(description="Technical analysis")
    critique: str = Field(description="Your thoughts on the initial answer")
    query: list[str] = Field(description="1-3 search queries to research improvements to address critiques of your current answer")

class AnswerQuestionRevise(BaseModel):
    """Schema for the answer"""
    analysis: str = Field(description="Technical analysis")
    critique: str = Field(description="Your thoughts on the initial answer")
    query: list[str] = Field(description="1-3 search queries to research improvements to address critiques of your current answer")
    references: list[str] = Field(description="List of references to improve the answer")

# --- Utility functions ---
def save_to_vectorstore(analysis_text: str, symbol: str, market_context: Optional[str]) -> str:
    now_utc = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "doc_id": doc_id,
         "symbol": symbol,
        "analysis_date_utc": now_utc.isoformat(),
        "market_context_summary": market_context[:200] if market_context else "N/A",
    }

    content_to_embed = f"Symbol: {symbol}\nMarket Context: {market_context}\n\nTechnical Analysis:\n{analysis_text}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content_to_embed, metadata=metadata)], ids=[doc_id])
        return f"Analysis saved to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Error saving analysis to vector store: {e}"

# --- Các Node của Graph ---
def generate_initial_response(state: ReflectionState) -> ReflectionState:
    symbol = state["symbol"]
    market_context = state["market_context"]
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a technical analysis expert in financial markets. Your task is to provide in-depth technical analysis for a stock symbol based on market context and data from analysis tools.

                Requirements:
                1. Use the `technical_indicator_analysis` tool to get data on technical indicators (MA, EMA, RSI, MACD, Bollinger Bands, ATR).
                2. Based on the tool data, identify potential price patterns (e.g., Doji, Double Top/Bottom) and short/long-term trends.
                3. Identify support and resistance zones based on Bollinger Bands or recent price levels from tool data.
                4. Assess market sentiment based on RSI and MACD.
                5. Provide short-term forecast (1-2 weeks) based on technical signals.
                6. Output format (maximum 300 words):
                - **Technical Indicator Summary**: Summarize MA, EMA, RSI, MACD, Bollinger Bands, ATR.
                - **Price Patterns and Trends**: Describe price patterns and trends.
                - **Support and Resistance**: Key price levels.
                - **Market Sentiment**: Based on RSI, MACD.
                - **Short-term Forecast**: Price trend prediction.
                7. Ensure analysis is concise, accurate, and aligned with the market context."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Provide technical analysis for {symbol}, based on the market context: {market_context}."""
            ),
        ]
    )
    
    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_context": market_context,
    }

    chain = prompt_template | llm_with_tools

    analysis_message = chain.invoke(prompt_input)

    if hasattr(analysis_message, 'tool_calls') and analysis_message.tool_calls:
        return {
            **state,
            "messages": messages + [analysis_message],
        }
        
    final_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a technical analysis expert in financial markets. Your task is to synthesize results from the technical analysis tool and create a structured response.

                Requirements:
                1. Synthesize results from the technical analysis tools to describe technical indicators.
                2. Provide comments on the analysis results.
                3. Structure your response according to this format:
                   - analysis: Provide comprehensive technical analysis based on the tool results
                   - critique: Offer thoughts on limitations or areas for improvement in the analysis
                   - query: Suggest 1-3 specific search queries to research improvements
                   - references: Include a list of reference types that would help improve the analysis
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """From the following technical analysis results, create a response and suggest search queries to improve the analysis for {symbol}:
                **Market Context:** {market_context}
                **Analysis Results:** {analysis}"""
            ),
        ]
    )
    
    prompt_input_final = {
        "messages": messages,
        "symbol": symbol,
        "market_context": market_context,
        "analysis": analysis_message.content,
    }
    chain_final = final_template | llm.with_structured_output(AnswerQuestionResponder)
        
    final_message = chain_final.invoke(prompt_input_final)
    
    analysis_content = final_message.analysis
    critique_content = final_message.critique
    query_content = final_message.query

    return {
        **state,
        "response": analysis_content,
        "critique": critique_content,
        "query": query_content,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Technical analysis for {symbol}:\n{analysis_content}")]
    }

def retrieve_historical(state: ReflectionState) -> ReflectionState:
    query = state["query"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not query:
        return {**state, "reflection_data": "No reflection query."}

    enhanced_query = f"{query} {symbol}"
    try:
        retrieved_docs = retriever.invoke(enhanced_query)
        insights = f"No history found related to: '{query}'."
        if retrieved_docs:
            insights_list = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                insights_list.append(
                    f"Insight {i+1} ({metadata.get('symbol', 'N/A')}, {metadata.get('analysis_date_utc', '?')}):\n{content_preview}"
                )
            insights = f"Found {len(retrieved_docs)} historical analyses related to '{query}':\n\n" + "\n\n".join(insights_list)
    except Exception as e:
        insights = f"Error retrieving history: {e}"

    return {
        **state,
        "reflection_data": insights,
        "messages": messages + [AIMessage(content=f"Historical query results:\n{insights}")]
    }

def revisor_analysis(state: ReflectionState) -> ReflectionState:
    response = state["response"]
    reflection_data = state["reflection_data"]
    critique = state["critique"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a technical analysis expert. Your task is to refine the technical analysis based on feedback, history, and market context.

                Requirements:
                1. Review the initial analysis, feedback (`critique`), and historical data (`reflection_data`) to improve the analysis.
                2. Address limitations mentioned in the `critique` (e.g., add price patterns, clearly identify support/resistance).
                3. Integrate information from `reflection_data` to increase accuracy (if available).
                4. Output format (maximum 300 words):
                - **Technical Indicator Summary**: Update based on new data.
                - **Price Patterns and Trends**: Clearly identify price patterns and trends.
                - **Support and Resistance**: Key price levels.
                - **Market Sentiment**: Based on RSI, MACD.
                - **Short-term Forecast**: Price trend prediction."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Refine the technical analysis for {symbol} based on:
                - Initial analysis: {response}
                - Feedback: {critique}
                - Historical data: {reflection_data}"""

            ),
        ]
    )
    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "response": response,
        "reflection_data": reflection_data,
        "critique": critique,
    }
    
    structured_llm = llm.with_structured_output(AnswerQuestionRevise)
    chain = prompt_template | structured_llm
    analysis_message = chain.invoke(prompt_input)
    
    
    refined_analysis_content = analysis_message.analysis
    refined_critique_content = analysis_message.critique
    refined_query_content = analysis_message.query
    refined_references_content = analysis_message.references
    
    return {
        **state,
        "response": refined_analysis_content,
        "critique": refined_critique_content,
        "query": refined_query_content,
        "references": refined_references_content,
        "reflection_iteration": iteration + 1,
        "messages": messages + [AIMessage(content=f"Refined analysis (Iter {iteration + 1}):\n{refined_analysis_content}")]
    }

def format_final_output(state: ReflectionState) -> ReflectionState:
    symbol = state["symbol"]
    final_analysis = state["response"]
    market_context = state["market_context"]
    messages = state["messages"]

    output = f"""
        # Technical Analysis for {symbol}
        **Market Context:**  
        {market_context}

        ## Detailed Analysis
        {final_analysis}
        """
    
    save_to_vectorstore(output, symbol, market_context)

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---
def should_continue_reflection(state: ReflectionState) -> Literal["retrieve_historical", "format_final_output"]:
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]

    if iteration <= max_reflections:
        return "retrieve_historical"
    return "format_final_output"

def build_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(ReflectionState)
    workflow.add_node("generate_initial_response", generate_initial_response)
    workflow.add_node("retrieve_historical", retrieve_historical)
    workflow.add_node("revisor_analysis", revisor_analysis)
    workflow.add_node("format_final_output", format_final_output)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "generate_initial_response")
    workflow.add_conditional_edges(
        "generate_initial_response",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "retrieve_historical"
        }
    )
    workflow.add_edge("tools", "generate_initial_response")
    workflow.add_conditional_edges(
        "revisor_analysis",
        should_continue_reflection,
        {
            "retrieve_historical": "retrieve_historical",
            "format_final_output": "format_final_output"
        }
    )
    workflow.add_edge("retrieve_historical", "revisor_analysis")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def low_level_agent(symbol: str, market_context: Optional[str], max_reflections: int = 2):
    initial_state = {
        "messages": [HumanMessage(content=f"Phân tích kỹ thuật {symbol}.")],
        "symbol": symbol,
        "market_context": market_context,
        "response": "",
        "query": None,
        "reflection_data": None,
        "reflection_iteration": 0,
        "max_reflections": max_reflections,
        "references": None,
        "final_output": None,
    }

    graph = build_workflow()
    result = graph.invoke(initial_state)
    return result.get("final_output", "Không có kết quả cuối cùng.")

graph = build_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    market_context = (
        f"Market Context for {symbol_to_analyze}:\n"
        f"- Overview: The general market has been moving sideways in the past few weeks.\n"
        f"- Sector: Technology is showing signs of accumulation.\n"
        f"- News: No recent significant news affecting the price."
    )
    max_reflections = 2

    result = low_level_agent(symbol_to_analyze, market_context, max_reflections)
    print(result)