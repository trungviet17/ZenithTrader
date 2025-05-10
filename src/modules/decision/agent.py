import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import pytz
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Khởi tạo LLM, Embeddings và Vector Store ---
llm = ChatOllama(model="cogito:3b")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0.2,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
embeddings = OllamaEmbeddings(model="cogito:3b")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./decision/qdrant_data"
COLLECTION_NAME = "decision_history"
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

# --- Định nghĩa State cho Decision Agent ---
class DecisionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_context: Optional[str]
    technical_analysis: Optional[str]
    improvement_data: Optional[str]
    trading_strategy: Optional[str]
    risk_assessment: Optional[str]
    final_decision: Optional[str]
    final_output: Optional[str]

class AnswerQuestion(BaseModel):
    """Schema cho câu trả lời quyết định giao dịch."""
    analysis: str = Field(description="Phân tích trạng thái thị trường hiện tại liên quan đến quyết định giao dịch.")
    reasoning: str = Field(description="Lý do đưa ra quyết định")
    action: str = Field(description="Mua, Bán, Giữ")

# --- Hàm tiện ích ---
def save_to_vectorstore(decision: str, symbol: str) -> str:
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
    }
    content = f"Trading Decision:\n{decision}"
    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved trading decision to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save trading decision: {e}"

# --- Các Node của Graph ---
def generate_trading_decision(state: DecisionState) -> DecisionState:
    """Node: Tạo quyết định giao dịch cuối cùng và lý do."""
    symbol = state["symbol"]
    market_context = state.get("market_context", "Không có dữ liệu thị trường.")
    technical_analysis = state.get("technical_analysis", "Không có phân tích kỹ thuật.")
    improvement_data = state.get("improvement_data", "Không có cải tiến.")
    trading_strategy = state.get("trading_strategy", "Không có chiến lược giao dịch.")
    risk_assessment = state.get("risk_assessment", "Không có đánh giá rủi ro.")
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia đưa ra quyết định giao dịch chứng khoán.
        Tổng hợp thông tin để đưa ra quyết định giao dịch (Mua, Bán, Giữ) cho mã chứng khoán.
        Dựa trên: Dữ liệu thị trường, phân tích kỹ thuật, cải tiến, chiến lược, và rủi ro.
        Định dạng: Phân tích ngắn gọn, lý do rõ ràng, hành động cụ thể (Mua/Bán/Giữ). Tối đa 250 từ."""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", """Đưa ra quyết định giao dịch cho {symbol}.
        **Dữ liệu Thị trường:** {market_context}
        **Phân tích Kỹ thuật:** {technical_analysis}
        **Cải tiến Quyết định:** {improvement_data}
        **Chiến lược Giao dịch:** {trading_strategy}
        **Đánh giá Rủi ro:** {risk_assessment}
        """),
    ])
    prompt_input = { 
        "messages": messages,
        "symbol": symbol,
        "market_context": market_context,
        "technical_analysis": technical_analysis,
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
        "risk_assessment": risk_assessment,
    }

    chain = prompt_template| llm.with_structured_output(AnswerQuestion)
    decision_content = chain.invoke(prompt_input)
    save_to_vectorstore(decision_content, symbol)

    return {
        **state,
        "final_decision": decision_content,
        "messages": messages + [AIMessage(content=f"Quyết định giao dịch cho {symbol}:\n{decision_content.analysis}\nLý do: {decision_content.reasoning}\nHành động: {decision_content.action}")],
    }

def format_final_output(state: DecisionState) -> DecisionState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    final_decision = state["final_decision"]
    messages = state["messages"]
    output = f"""
            Quyết định giao dịch cho {symbol}:\n
            - Phân tích: {final_decision.analysis}
            - Lý do: {final_decision.reasoning}
            - Hành động: {final_decision.action}
            """
    return {
        **state,
        "final_output": output,
        "messages": messages + [AIMessage(content=output)],
    }

# --- Xây dựng Graph ---
def build_decision_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(DecisionState)
    workflow.add_node("generate_trading_decision", generate_trading_decision)
    workflow.add_node("format_final_output", format_final_output)
    workflow.add_edge(START, "generate_trading_decision")
    workflow.add_edge("generate_trading_decision", "format_final_output")
    workflow.add_edge("format_final_output", END)
    return workflow.compile()

# --- Hàm chính ---
def decision_agent(
    symbol: str,
    market_context: Optional[str] = None,
    technical_analysis: Optional[str] = None,
    improvement_data: Optional[str] = None,
    trading_strategy: Optional[str] = None,
    risk_assessment: Optional[str] = None,
) -> str:
    """Chạy agent quyết định giao dịch."""
    if not symbol:
        return "Lỗi: Mã chứng khoán không được để trống."
    initial_state = {
        "messages": [HumanMessage(content=f"Đưa ra quyết định giao dịch cho {symbol}.")],
        "symbol": symbol,
        "market_context": market_context,
        "technical_analysis": technical_analysis,
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
        "risk_assessment": risk_assessment,
        "final_decision": None,
        "final_output": None,
    }
    graph = build_decision_workflow()
    result = graph.invoke(initial_state)
    return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")

graph = build_decision_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    market_context = (
        f"Bối cảnh Thị trường cho {symbol_to_analyze}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: Công nghệ đang có dấu hiệu tích lũy.\n"
        f"- Tin tức: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    technical_analysis = (
        f"Phân tích Kỹ thuật cho {symbol_to_analyze}:\n"
        f"- RSI: 65, gần vùng overbought.\n"
        f"- MACD: Bullish crossover, xu hướng tăng.\n"
        f"- Kháng cự: 155, hỗ trợ: 145."
    )
    improvement_data = (
        f"Cải tiến Quyết định cho {symbol_to_analyze}:\n"
        f"- Tránh bán khi RSI gần overbought nhưng MACD vẫn bullish.\n"
        f"- Tín hiệu kháng cự tại 155 không đáng tin cậy trong xu hướng tăng mạnh.\n"
        f"- Đề xuất: Tăng tỷ trọng khi giá phá vỡ kháng cự với khối lượng cao."
    )
    trading_strategy = (
        f"Chiến lược Giao dịch cho {symbol_to_analyze}:\n"
        f"- Chiến lược: Momentum Trading\n"
        f"- Triển khai: Mua khi giá phá vỡ kháng cự 155 với khối lượng cao, bán khi MACD chuyển bearish.\n"
        f"- Lý do: Xu hướng tăng mạnh, MACD bullish, khối lượng giao dịch tăng."
    )
    risk_assessment = (
        f"Đánh giá Rủi ro cho {symbol_to_analyze}:\n"
        f"- Biến động: Cao (30% hàng năm), rủi ro giá giảm ngắn hạn.\n"
        f"- Thanh khoản: Tốt, khối lượng giao dịch ổn định.\n"
        f"- Rủi ro chiến lược: Mua gần kháng cự 155 có thể rủi ro nếu không phá vỡ.\n"
        f"- Chiến lược giảm thiểu: Đặt dừng lỗ tại 145, phân bổ vốn không quá 20% danh mục."
    )
    result = decision_agent(
        symbol_to_analyze,
        market_context,
        technical_analysis,
        improvement_data,
        trading_strategy,
        risk_assessment,
    )
    print(result)