import yfinance as yf
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from uuid import uuid4

# Langchain và Qdrant imports
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Khởi tạo LLM, Embeddings và Vector Store ---
llm = ChatOllama(model="cogito:3b")
embeddings = OllamaEmbeddings(model="cogito:3b")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./high_level/qdrant_data"
COLLECTION_NAME = "high_level_reflection_history"
client = QdrantClient(path=QDRANT_PATH)

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Tạo collection Qdrant nếu nó chưa tồn tại."""
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

# --- Định nghĩa State cho High-Level Reflection Agent ---
class HighLevelReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_data: Optional[str]           # Dữ liệu thị trường từ Market Agent
    technical_analysis: Optional[str]     # Phân tích kỹ thuật từ TA Agent
    trade_history: Optional[List[Dict]]   # Lịch sử giao dịch
    current_decision_analysis: str        # Phân tích quyết định hiện tại
    reflection_query: Optional[str]       # Truy vấn reflection
    historical_insights: Optional[str]    # Thông tin lịch sử từ vector store
    reflection_iteration: int             # Số lần lặp reflection
    max_reflections: int                  # Số lần lặp tối đa
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Tools cho High-Level Reflection ---

@tool
def fetch_market_data(symbol: str) -> str:
    """
    Lấy bối cảnh thị trường chung và tin tức liên quan đến cổ phiếu.
    
    Args:
        symbol: Mã cổ phiếu (ticker symbol).
    
    Returns:
        Chuỗi mô tả bối cảnh thị trường và tin tức liên quan.
    """
    return (
        f"Bối cảnh Thị trường Mô phỏng cho {symbol}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: [Ngành của {symbol}] đang có dấu hiệu tích lũy.\n"
        f"- Tin tức {symbol}: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    
@tool
def fetch_TA_data(symbol: str) -> str:
    """
    Lấy phân tích kỹ thuật cho cổ phiếu.
    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        
    Returns:
        Chuỗi mô tả phân tích kỹ thuật cho cổ phiếu.
    
    """
    return (
        f"Phân tích Kỹ thuật Mô phỏng cho {symbol}:\n"
        f"- Xu hướng: Bullish (MA20 > MA50).\n"
        f"- RSI: 65 (Neutral, gần Overbought).\n"
        f"- MACD: Bullish Crossover.\n"
        f"- Hỗ trợ: 145; Kháng cự: 155.\n"
        f"- Dự báo ngắn hạn: Tăng nhẹ trong vài tuần tới."
    )

@tool
def fetch_trade_history(symbol: str) -> List[Dict]:
    """
    Lấy lịch sử giao dịch giả lập cho cổ phiếu.
    
    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        
    Returns:
        Danh sách các giao dịch với thông tin chi tiết.
    """
    # Mô phỏng lịch sử giao dịch trong 3 tháng
    return [
        {
            "trade_id": str(uuid4()),
            "symbol": symbol,
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
            "symbol": symbol,
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

# --- Hàm tiện ích ---
def save_analysis(analysis: str, symbol: str) -> str:
    """Lưu phân tích hoặc insight vào vector store."""
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
    }
    content = analysis

    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved reflection to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save reflection to vector store: {e}"

# --- Các Node của Graph ---

def get_initial_data(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Lấy dữ liệu thị trường, phân tích kỹ thuật, lịch sử giao dịch và bối cảnh thị trường."""
    symbol = state["symbol"]
    messages = state["messages"]

    # Lấy dữ liệu thị trường
    market_data = fetch_market_data.invoke({"symbol": symbol})

    # Lấy phân tích kỹ thuật
    technical_analysis = fetch_TA_data.invoke({"symbol": symbol})

    # Lấy lịch sử giao dịch
    trade_history = fetch_trade_history.invoke({"symbol": symbol})

    return {
        **state,
        "market_data": market_data,
        "technical_analysis": technical_analysis,
        "trade_history": trade_history,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Đã lấy dữ liệu thị trường, phân tích kỹ thuật, và lịch sử giao dịch cho {symbol}.")]
    }

def generate_analysis(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Tạo phân tích quyết định giao dịch ban đầu."""
    market_data = state["market_data"]
    technical_analysis = state["technical_analysis"]
    trade_history = state["trade_history"]
    symbol = state["symbol"]
    messages = state["messages"]

    # Format lịch sử giao dịch
    trade_summary = "\n".join([
        f"- Trade {t['date']} ({t['action']} @ {t['price']}): {t['reason']} Outcome: {t['outcome']} (P/L: {t['profit_loss']}). Analysis: {t['analysis']}"
        for t in trade_history
    ])

    prompt = f"""Bạn là chuyên gia phân tích giao dịch chứng khoán, hãy đánh giá và cải thiện các quyết định giao dịch cho {symbol}. 
                Dựa trên thông tin sau, đưa ra phân tích ban đầu về các quyết định giao dịch:
                
                **Bối cảnh Thị trường:**
                {market_data}

                **Phân tích Kỹ thuật:**
                {technical_analysis}

                **Lịch sử Giao dịch:**
                {trade_summary}

                **Yêu cầu:**
                1. Đánh giá tính đúng/sai của các quyết định giao dịch trong lịch sử (dựa trên phân tích kỹ thuật và outcome).
                2. Phân tích lý do tại sao các quyết định đúng hoặc sai.
                3. Đề xuất cải thiện cho các quyết định giao dịch trong tương lai.
                4. Đưa ra khuyến nghị giao dịch hiện tại dựa trên phân tích kỹ thuật và bối cảnh thị trường.

                **Phân tích ban đầu của bạn:**
                """

    analysis_message = llm.invoke(prompt)
    analysis_content = analysis_message.content

    return {
        **state,
        "current_decision_analysis": analysis_content,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Phân tích quyết định ban đầu cho {symbol}:\n{analysis_content}")]
    }

def generate_reflection_query(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Tạo câu hỏi reflection để cải thiện phân tích quyết định."""
    current_analysis = state["current_decision_analysis"]
    symbol = state["symbol"]
    trade_history = state["trade_history"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]

    if not current_analysis or iteration >= max_reflections:
        return {**state, "reflection_query": None}

    trade_summary = "\n".join([
        f"- {t['action']} @ {t['date']}: {t['outcome']} (P/L: {t['profit_loss']})"
        for t in trade_history
    ])

    prompt = f"""Xem xét phân tích quyết định giao dịch cho {symbol}:
            **Phân tích Hiện tại:**
            {current_analysis}

            **Lịch sử Giao dịch Tóm tắt:**
            {trade_summary}

            **Yêu cầu:**
            Để cải thiện phân tích quyết định, đặt 1-2 câu hỏi cụ thể cần kiểm tra trong lịch sử (các quyết định giao dịch trước đây của {symbol}). 
            Tập trung vào điểm không chắc chắn, sai lầm trong quyết định, hoặc cơ hội cải thiện.

            **Ví dụ:**
            - "Khi RSI ở mức tương tự, các quyết định mua/bán trước đây có thành công không?"
            - "Tín hiệu kháng cự tại 155 có thường xuyên bị phá vỡ trong lịch sử không?"
            - "Quyết định bán dựa trên RSI overbought có đáng tin cậy không?"

            **Câu hỏi reflection của bạn:**
            """

    query_message = llm.invoke(prompt)
    query_content = query_message.content

    return {
        **state,
        "reflection_query": query_content,
        "messages": messages + [AIMessage(content=f"Câu hỏi Reflection (Iter {iteration}, 3mo):\n{query_content}")]
    }

def retrieve_historical(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Truy vấn vector store lấy thông tin lịch sử."""
    query = state["reflection_query"]
    messages = state["messages"]

    if not query:
        return {**state, "historical_insights": "Không có truy vấn reflection."}

    enhanced_query = f"{query}"
    try:
        retrieved_docs = retriever.invoke(enhanced_query)
        insights = f"Không tìm thấy lịch sử liên quan đến: '{query}'."
        if retrieved_docs:
            insights_list = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                insights_list.append(
                    f"Insight {i+1} ({metadata.get('symbol', 'N/A')}, {metadata.get('timeframe','?')}, {metadata.get('type', '?')}):\n'{content_preview}'"
                )
            insights = f"Tìm thấy {len(retrieved_docs)} ghi chú lịch sử liên quan '{query}':\n\n" + "\n\n".join(insights_list)

    except Exception as e:
        insights = f"Lỗi khi truy xuất lịch sử: {e}"

    return {
        **state,
        "historical_insights": insights,
        "messages": messages + [AIMessage(content=f"Kết quả truy vấn lịch sử:\n{insights}")]
    }

def refine_decision_analysis(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Tinh chỉnh phân tích quyết định dựa trên thông tin lịch sử."""
    current_analysis = state["current_decision_analysis"]
    historical_insights = state["historical_insights"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]

    if not historical_insights or iteration >= max_reflections:
        return {**state, "reflection_iteration": iteration + 1}

    prompt = f"""Cải thiện phân tích quyết định giao dịch cho {symbol}.

            **Phân tích Hiện tại:**
            {current_analysis}

            **Thông tin Lịch sử:**
            {historical_insights}

            **Yêu cầu:**
            1. Xem xét thông tin lịch sử và so sánh với phân tích hiện tại.
            2. Cập nhật đánh giá về các quyết định giao dịch (tại sao đúng/sai, cải thiện).
            3. Đưa ra khuyến nghị giao dịch hiện tại dựa trên dữ liệu mới.
            4. Nếu lịch sử không hữu ích, ghi nhận và điều chỉnh nhẹ.

            **Phân tích đã tinh chỉnh:**
            """

    refined_analysis_message = llm.invoke(prompt)
    refined_analysis_content = refined_analysis_message.content

    return {
        **state,
        "current_decision_analysis": refined_analysis_content,
        "reflection_iteration": iteration + 1,
        "messages": messages + [AIMessage(content=f"Phân tích quyết định đã tinh chỉnh (Iter {iteration + 1}):\n{refined_analysis_content}")]
    }

def format_final_output(state: HighLevelReflectionState) -> HighLevelReflectionState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    final_analysis = state["current_decision_analysis"]
    messages = state["messages"]

    if not final_analysis:
        output = f"Không thể hoàn thành phân tích quyết định cho {symbol} do thiếu dữ liệu."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    save_analysis(final_analysis, symbol)

    output = f"""# Phân tích Quyết định Giao dịch {symbol}:
                {final_analysis}
            """

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---

def reflection_condition(state: HighLevelReflectionState) -> Literal["retrieve_historical", "format_final_output"]:
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]
    query = state["reflection_query"]

    if query and iteration < max_reflections:
        return "retrieve_historical"
    
    return "format_final_output"

def build_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(HighLevelReflectionState)

    workflow.add_node("get_initial_data", get_initial_data)
    workflow.add_node("generate_analysis", generate_analysis)
    workflow.add_node("generate_reflection_query", generate_reflection_query)
    workflow.add_node("retrieve_historical", retrieve_historical)
    workflow.add_node("refine_decision_analysis", refine_decision_analysis)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "get_initial_data")
    workflow.add_edge("get_initial_data", "generate_analysis")
    workflow.add_edge("generate_analysis", "generate_reflection_query")
    workflow.add_conditional_edges(
        "generate_reflection_query",
        reflection_condition,
        {
            "retrieve_historical": "retrieve_historical",
            "format_final_output": "format_final_output"
        }
    )
    workflow.add_edge("retrieve_historical", "refine_decision_analysis")
    workflow.add_edge("refine_decision_analysis", "generate_reflection_query")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

graph = build_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    max_reflections = 1

    graph = build_workflow()

    initial_state = {
        "messages": [HumanMessage(content=f"Phân tích quyết định giao dịch {symbol_to_analyze}.")],
        "symbol": symbol_to_analyze,
        "max_reflections": max_reflections,
        "reflection_iteration": 0,
        "current_decision_analysis": "",
        "reflection_query": None,
        "historical_insights": None,
        "market_data": None,
        "technical_analysis": None,
        "trade_history": None,
        "final_output": None,
    }

    print(f"\n--- Bắt đầu Workflow Phân tích Quyết định Giao dịch (3 tháng) cho {symbol_to_analyze} ---")
    final_result_state = graph.invoke(initial_state)
    print(f"\n--- Workflow Phân tích Quyết định Giao dịch cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Phân tích Cuối cùng ---")
    if final_result_state.get("final_output"):
        print(final_result_state["final_output"])