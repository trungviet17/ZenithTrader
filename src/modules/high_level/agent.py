from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from uuid import uuid4
import json
from pydantic import BaseModel, Field
import os

from langchain_core.tools import tool
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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

# --- Khởi tạo LLM, Embeddings và Vector Store ---
llm = LLM.get_gemini_llm(model_index = 3)
embeddings = LLM.get_gemini_embedding()


# --- Thiết lập Qdrant ---
QDRANT_PATH = "./high_level/qdrant_data"
COLLECTION_NAME = "high_level_reflection_history"
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

# --- Định nghĩa State ---
class HighLevelReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_data: Optional[str]
    technical_analysis: Optional[str]
    trade_history: Optional[List[Dict]]
    critique: Optional[str]
    response: Optional[str]
    query: Optional[str]
    reflection_data: Optional[str]
    reflection_iteration: Optional[int]
    max_reflections: int
    final_output: Optional[str]

# --- Tools Phân tích Giao dịch ---
@tool
def get_current_time() -> str:
    """Lấy thời gian hiện tại"""
    return datetime.now(pytz.UTC).isoformat()

tools = [get_current_time]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

class AnswerQuestionResponder(BaseModel):
    """Schema cho câu trả lời"""
    analysis: str = Field(description="Phân tích kỹ thuật")
    critique: str = Field(description="Suy nghĩ của bạn về câu trả lời ban đầu")
    query: list[str] = Field(description="1-3 truy vấn tìm kiếm để nghiên cứu cải tiến nhằm giải quyết lời chỉ trích về câu trả lời hiện tại của bạn")

class AnswerQuestionRevise(BaseModel):
    """Schema cho câu trả lời"""
    analysis: str = Field(description="Phân tích kỹ thuật")
    critique: str = Field(description="Suy nghĩ của bạn về câu trả lời ban đầu")
    query: list[str] = Field(description="1-3 truy vấn tìm kiếm để nghiên cứu cải tiến nhằm giải quyết lời chỉ trích về câu trả lời hiện tại của bạn")
    references: list[str] = Field(description="Danh sách các tài liệu tham khảo để cải tiến câu trả lời")

# --- Hàm tiện ích ---
def save_to_vectorstore(analysis_text: str, symbol: str, market_data: Optional[str]) -> str:
    now_utc = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "doc_id": doc_id,
        "symbol": symbol,
        "analysis_date_utc": now_utc.isoformat(),
        "market_data_summary": market_data[:200] if market_data else "N/A",
    }
    content_to_embed = f"Mã CP: {symbol}\nBối cảnh thị trường: {market_data}\n\nPhân tích giao dịch:\n{analysis_text}"
    try:
        vector_store.add_documents(documents=[Document(page_content=content_to_embed, metadata=metadata)], ids=[doc_id])
        return f"Đã lưu phân tích vào vector store (ID: {doc_id})"
    except Exception as e:
        return f"Lỗi khi lưu phân tích vào vector store: {e}"

# --- Các Node của Graph ---
def generate_initial_response(state: HighLevelReflectionState) -> HighLevelReflectionState:
    symbol = state["symbol"]
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    trade_history = state["trade_history"]
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a stock trading analysis expert, capable of evaluating and optimizing trading decisions.

                Requirements:
                1. Analyze past trades of the stock based on:
                   - Market context.
                   - Technical analysis (support, resistance, RSI, MACD, etc.).
                   - Trading history (date, action, price, reason, result).
                2. Evaluate each trade:
                   - Correct/incorrect decision? Why (based on technical signals, market)?
                   - If incorrect, suggest improvements (change entry/exit points, volume, timing).
                3. Profit optimization:
                   - Simulate alternative trading scenarios to increase profit.
                4. Summarize lessons learned:
                   - From successes/mistakes, derive lessons applicable to the current context.
                5. Provide current trading recommendations based on technical and market analysis.
                6. Output format:
                   - Summary of market context and technical analysis.
                   - Evaluation of each trade (correct/incorrect, reasons, improvements).
                   - Alternative scenarios to optimize profit.
                   - Lessons learned.
                   - Current trading recommendations.
                   - Maximum 400 words, clear and concise."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Analyze trading decisions for stock symbol {symbol}.
                Market context: {market_data}
                Technical analysis: {technical_analysis}
                Trading history: {trade_history}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data,
        "technical_analysis": technical_analysis,
        "trade_history": trade_history,
    }
    chain = prompt_template | llm.with_structured_output(AnswerQuestionResponder)

    analysis_message = chain.invoke(prompt_input)

    if hasattr(analysis_message, 'tool_calls') and analysis_message.tool_calls:
        return {
            **state,
            "messages": messages + [analysis_message],
        }
    analysis_content = analysis_message.analysis
    critique_content = analysis_message.critique
    query_content = analysis_message.query
    
    return {
        **state,
        "response": analysis_content,
        "critique": critique_content,
        "query": query_content,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Phân tích quyết định giao dịch cho {symbol}:\n{analysis_content}")]
    }

def retrieve_historical(state: HighLevelReflectionState) -> HighLevelReflectionState:
    query = state["query"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not query:
        return {**state, "reflection_data": "Không có truy vấn reflection."}

    enhanced_query = f"{query} {symbol}"
    try:
        retrieved_docs = retriever.invoke(enhanced_query)
        insights = f"Không tìm thấy lịch sử liên quan đến: '{query}'."
        if retrieved_docs:
            insights_list = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                insights_list.append(
                    f"Insight {i+1} ({metadata.get('symbol', 'N/A')}, {metadata.get('analysis_date_utc', '?')}):\n{content_preview}"
                )
            insights = f"Tìm thấy {len(retrieved_docs)} phân tích lịch sử liên quan '{query}':\n\n" + "\n\n".join(insights_list)
    except Exception as e:
        insights = f"Lỗi khi truy xuất lịch sử: {e}"

    return {
        **state,
        "reflection_data": insights,
        "messages": messages + [AIMessage(content=f"Kết quả truy vấn lịch sử:\n{insights}")]
    }

def refine_analysis(state: HighLevelReflectionState) -> HighLevelReflectionState:
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
                """You are a stock trading analysis expert, capable of evaluating and optimizing trading decisions.

                Requirements:
                1. Refine trading decision analysis based on user feedback.
                2. Evaluate the accuracy, logic, and practicality of the analysis.
                3. Provide 1-2 strengths and 1-2 weaknesses in the analysis.
                4. Create query questions to search history for improving the analysis.
                5. Output format:
                   - Refined analysis.
                   - Evaluation (strengths/weaknesses).
                   - History search query questions.
                   - Maximum 400 words for analysis, 100 words for evaluation, 50 words for questions."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Refine the trading decision analysis for stock symbol {symbol}.
                Initial analysis: {response}
                Evaluation: {critique}
                History: {reflection_data}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "response": response,
        "critique": critique,
        "reflection_data": reflection_data,
    }
    chain = prompt_template | llm.with_structured_output(AnswerQuestionRevise)
    refined_message = chain.invoke(prompt_input)
    refined_analysis_content = refined_message.analysis
    refined_critique_content = refined_message.critique
    refined_query_content = refined_message.query
    refined_references_content = refined_message.references

    return {
        **state,
        "response": refined_analysis_content,
        "critique": refined_critique_content,
        "query": refined_query_content,
        "reflection_data": refined_references_content,
        "reflection_iteration": iteration + 1,
        "messages": messages + [AIMessage(content=f"Phân tích đã tinh chỉnh (Iter {iteration + 1}):\n{refined_analysis_content}")]
    }

def format_final_output(state: HighLevelReflectionState) -> HighLevelReflectionState:
    symbol = state["symbol"]
    final_analysis = state["response"]
    market_data = state["market_data"]
    messages = state["messages"]

    if not final_analysis:
        output = f"Không thể hoàn thành phân tích quyết định giao dịch cho {symbol} do thiếu dữ liệu."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    save_to_vectorstore(final_analysis, symbol, market_data)

    output = f"""
        # Phân tích Quyết định Giao dịch {symbol}
        **Bối cảnh Thị trường:**  
        {market_data}

        ## Phân tích Chi tiết
        {final_analysis}
        """

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---
def should_continue_reflection(state: HighLevelReflectionState) -> Literal["retrieve_historical", "format_final_output"]:
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]
    if iteration < max_reflections:
        return "retrieve_historical"
    return "format_final_output"

def build_workflow():
    workflow = StateGraph(HighLevelReflectionState)
    workflow.add_node("generate_initial_response", generate_initial_response)
    workflow.add_node("retrieve_historical", retrieve_historical)
    workflow.add_node("refine_analysis", refine_analysis)
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
    workflow.add_edge("retrieve_historical", "refine_analysis")
    workflow.add_conditional_edges(
        "refine_analysis",
        should_continue_reflection,
        {
            "retrieve_historical": "retrieve_historical",
            "format_final_output": "format_final_output"
        }
    )
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def high_level_agent(symbol: str, market_data: Optional[str], technical_analysis: Optional[str], trade_history: Optional[List[Dict]], max_reflections: int = 2):
    initial_state = {
        "messages": [HumanMessage(content=f"Phân tích quyết định giao dịch {symbol}.")],
        "symbol": symbol,
        "market_data": market_data,
        "technical_analysis": technical_analysis,
        "trade_history": trade_history,
        "response": "",
        "critique": None,
        "query": None,
        "reflection_data": None,
        "reflection_iteration": 0,
        "max_reflections": max_reflections,
        "final_output": None,
    }

    graph = build_workflow()
    result = graph.invoke(initial_state)
    return result.get("final_output", "Không có kết quả phân tích.")

graph = build_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    max_reflections = 2
    sample_market_data = (
        f"Bối cảnh Thị trường cho {symbol_to_analyze}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: Công nghệ đang có dấu hiệu tích lũy.\n"
        f"- Tin tức: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    sample_technical_analysis = (
        f"Phân tích Kỹ thuật cho {symbol_to_analyze}:\n"
        f"- Xu hướng: Bullish (MA20 > MA50).\n"
        f"- RSI: 65 (Neutral, gần Overbought).\n"
        f"- MACD: Bullish Crossover.\n"
        f"- Hỗ trợ: 145; Kháng cự: 155.\n"
        f"- Dự báo ngắn hạn: Tăng nhẹ trong vài tuần tới."
    )
    sample_trade_history = [
        {
            "trade_id": str(uuid4()),
            "symbol": symbol_to_analyze,
            "date": "2025-02-15",
            "action": "Buy",
            "price": 150.0,
            "quantity": 100,
            "reason": "RSI cho tín hiệu oversold, MACD crossover tích cực.",
            "outcome": "Profit",
            "profit_loss": 500.0,
            "analysis": "Quyết định đúng do giá tăng sau tín hiệu kỹ thuật."
        },
        {
            "trade_id": str(uuid4()),
            "symbol": symbol_to_analyze,
            "date": "2025-03-10",
            "action": "Sell",
            "price": 155.0,
            "quantity": 100,
            "reason": "Giá chạm kháng cự mạnh tại 155, RSI overbought.",
            "outcome": "Loss",
            "profit_loss": -200.0,
            "analysis": "Quyết định sai do giá tiếp tục tăng sau khi bán."
        }
    ]

    result = high_level_agent(
        symbol_to_analyze,
        sample_market_data,
        sample_technical_analysis,
        sample_trade_history,
        max_reflections
    )
    print(result)