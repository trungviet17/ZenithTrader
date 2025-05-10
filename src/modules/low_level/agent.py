import os
import requests
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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# --- Khởi tạo LLM, Embeddings và Vector Store ---
# llm = ChatOllama(model="cogito:3b")
embeddings = OllamaEmbeddings(model="cogito:3b")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

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
def technical_indicator_analysis(symbol: str, interval: str = "1day", outputsize: int = 30) -> Dict[str, str]:
    """
    Phân tích các mẫu dựa trên các chỉ số kĩ thuật của mã cổ phiếu.
    Args:
        symbol (str): Mã chứng khoán cần phân tích. (ví dụ: "AAPL").
        interval (str): Khoảng thời gian cho dữ liệu (mặc định là 1 ngày) (ví dụ: "8h", "1day", "1week", "1month").
        outputsize (int): Số lượng dữ liệu cần lấy (mặc định là 30).
    Returns:
        dict: Kết quả phân tích kỹ thuật dưới dạng từ điển.
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
                start_date="2025-01-01",
                end_date="2025-03-01"
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
            trend = "tăng" if ma_recent > ma_prev else "giảm"
            analysis["MA"] = f"MA có xu hướng {trend} từ {ma_prev:.2f} lên {ma_recent:.2f}."

        # EMA trend
        if "ema" in df.columns and len(df["ema"].dropna()) >= 5:
            ema_recent = df["ema"].dropna().iloc[-1]
            ema_prev = df["ema"].dropna().iloc[-5]
            trend = "tăng" if ema_recent > ema_prev else "giảm"
            analysis["EMA"] = f"EMA có xu hướng {trend} từ {ema_prev:.2f} lên {ema_recent:.2f}."

        # RSI
        if "rsi" in df.columns:
            rsi = df["rsi"].dropna().iloc[-1]
            if rsi > 70:
                status = "quá mua (có thể điều chỉnh giảm)"
            elif rsi < 30:
                status = "quá bán (có thể phục hồi)"
            else:
                status = "trung tính"
            analysis["RSI"] = f"RSI hiện tại là {rsi:.2f}, trạng thái {status}."

        # MACD
        if {"macd", "macd_signal", "macd_hist"}.issubset(df.columns):
            last = df.dropna(subset=["macd", "macd_signal", "macd_hist"]).iloc[-1]
            signal = "bullish" if last["macd"] > last["macd_signal"] else "bearish"
            analysis["MACD"] = (
                f"MACD: {last['macd']:.2f}, tín hiệu: {last['macd_signal']:.2f}, "
                f"histogram: {last['macd_hist']:.2f} → tín hiệu {signal}."
            )

        # Bollinger Bands
        if {"upper_band", "middle_band", "lower_band"}.issubset(df.columns):
            bb = df.dropna(subset=["upper_band", "middle_band", "lower_band"]).iloc[-1]
            spread = float(bb["upper_band"]) - float(bb["lower_band"])
            analysis["BollingerBands"] = (
                f"Dải Bollinger có biên độ {spread:.2f} điểm (Upper: {bb['upper_band']}, Lower: {bb['lower_band']})."
            )

        # ATR
        if "atr" in df.columns:
            atr = df["atr"].dropna().iloc[-1]
            analysis["ATR"] = f"Chỉ báo ATR là {atr:.2f}, biểu thị mức biến động trung bình gần đây."

        return analysis

    except Exception as e:
        return {"error": str(e)}
    
    
# --- Định nghĩa các công cụ --- 
tools = [technical_indicator_analysis]
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
def save_to_vectorstore(analysis_text: str, symbol: str, market_context: Optional[str]) -> str:
    now_utc = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "doc_id": doc_id,
         "symbol": symbol,
        "analysis_date_utc": now_utc.isoformat(),
        "market_context_summary": market_context[:200] if market_context else "N/A",
    }

    content_to_embed = f"Mã CP: {symbol}\nBối cảnh thị trường: {market_context}\n\nPhân tích kỹ thuật:\n{analysis_text}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content_to_embed, metadata=metadata)], ids=[doc_id])
        return f"Đã lưu phân tích vào vector store (ID: {doc_id})"
    except Exception as e:
        return f"Lỗi khi lưu phân tích vào vector store: {e}"

# --- Các Node của Graph ---
def generate_initial_response(state: ReflectionState) -> ReflectionState:
    symbol = state["symbol"]
    market_context = state["market_context"]
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là một chuyên gia phân tích kỹ thuật thị trường tài chính. Nhiệm vụ của bạn là cung cấp phân tích kỹ thuật chuyên sâu cho mã chứng khoán dựa trên bối cảnh thị trường và dữ liệu từ công cụ phân tích.

                Yêu cầu:
                1. Sử dụng công cụ `technical_indicator_analysis` để lấy dữ liệu các chỉ báo kỹ thuật (MA, EMA, RSI, MACD, Bollinger Bands, ATR).
                2. Dựa trên dữ liệu từ công cụ, nhận diện các mẫu giá tiềm năng (ví dụ: Doji, Double Top/Bottom) và xu hướng ngắn hạn/dài hạn.
                3. Xác định các vùng hỗ trợ, kháng cự dựa trên Bollinger Bands hoặc các mức giá gần đây từ dữ liệu công cụ.
                4. Đánh giá tâm lý thị trường dựa trên RSI và MACD.
                5. Đưa ra dự báo ngắn hạn (1-2 tuần) dựa trên các tín hiệu kỹ thuật.
                6. Định dạng đầu ra (tối đa 300 từ):
                - **Tóm tắt chỉ báo kỹ thuật**: Tổng hợp MA, EMA, RSI, MACD, Bollinger Bands, ATR.
                - **Mẫu giá và xu hướng**: Mô tả mẫu giá và xu hướng.
                - **Hỗ trợ, kháng cự**: Các mức giá quan trọng.
                - **Tâm lý thị trường**: Dựa trên RSI, MACD.
                - **Dự báo ngắn hạn**: Dự đoán xu hướng giá.
                7. Đảm bảo phân tích ngắn gọn, chính xác và phù hợp với bối cảnh thị trường."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Phân tích kỹ thuật cho mã chứng khoán {symbol}, dựa trên bối cảnh thị trường: {market_context}."""
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
                """Bạn là một chuyên gia phân tích kỹ thuật thị trường tài chính. Nhiệm vụ của bạn là tổng hợp kết quả từ công cụ phân tích kỹ thuật và tạo câu trả lời có cấu trúc.

                Yêu cầu:
                1. Tổng hợp kết quả từ công cụ `analyze_patterns_trends` để mô tả các chỉ báo kỹ thuật.
                2. Đưa ra nhận xét về kết quả phân tích.
                3. Đề xuất 1-3 truy vấn tìm kiếm để cải thiện phân tích.
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Từ kết quả phân tích kỹ thuật sau, tạo phản hồi và đề xuất truy vấn tìm kiếm để cải thiện phân tích cho mã chứng khoán {symbol}:
                **Bối cảnh Thị trường:** {market_context}
                **Kết quả phân tích:** {analysis}"""
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
        "messages": messages + [AIMessage(content=f"Phân tích kỹ thuật cho {symbol}:\n{analysis_content}")]
    }

def retrieve_historical(state: ReflectionState) -> ReflectionState:
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
                """Bạn là một chuyên gia phân tích kỹ thuật. Nhiệm vụ của bạn là tinh chỉnh phân tích kỹ thuật dựa trên phản hồi, lịch sử, và bối cảnh thị trường.

                Yêu cầu:
                1. Xem xét phân tích ban đầu, phản hồi (`critique`), và dữ liệu lịch sử (`reflection_data`) để cải thiện phân tích.
                2. Giải quyết các hạn chế được nêu trong `critique` (ví dụ: bổ sung mẫu giá, xác định rõ hỗ trợ/kháng cự).
                3. Tích hợp thông tin từ `reflection_data` để tăng độ chính xác (nếu có).
                4. Định dạng đầu ra (tối đa 300 từ):
                - **Tóm tắt chỉ báo kỹ thuật**: Cập nhật dựa trên dữ liệu mới.
                - **Mẫu giá và xu hướng**: Xác định rõ mẫu giá và xu hướng.
                - **Hỗ trợ, kháng cự**: Các mức giá quan trọng.
                - **Tâm lý thị trường**: Dựa trên RSI, MACD.
                - **Dự báo ngắn hạn**: Dự đoán xu hướng giá."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Tinh chỉnh phân tích kỹ thuật cho mã chứng khoán {symbol} dựa trên:
                - Phân tích ban đầu: {response}
                - Phản hồi: {critique}
                - Dữ liệu lịch sử: {reflection_data}"""
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
        "messages": messages + [AIMessage(content=f"Phân tích đã tinh chỉnh (Iter {iteration + 1}):\n{refined_analysis_content}")]
    }

def format_final_output(state: ReflectionState) -> ReflectionState:
    symbol = state["symbol"]
    final_analysis = state["response"]
    market_context = state["market_context"]
    messages = state["messages"]

    output = f"""
        # Phân tích Kỹ thuật {symbol}
        **Bối cảnh Thị trường:**  
        {market_context}

        ## Phân tích Chi tiết
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
        f"Bối cảnh Thị trường cho {symbol_to_analyze}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: Công nghệ đang có dấu hiệu tích lũy.\n"
        f"- Tin tức: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    max_reflections = 2

    result = low_level_agent(symbol_to_analyze, market_context, max_reflections)
    print(result)