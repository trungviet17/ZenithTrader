import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import pytz

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
# llm = ChatOllama(model="cogito:3b")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embeddings = OllamaEmbeddings(model="cogito:3b")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./risk_management/qdrant_data"
COLLECTION_NAME = "risk_management_history"
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

# --- Định nghĩa State cho Risk Management Agent ---
class RiskManagementState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_data: Optional[Dict]           # Thông tin thị trường
    improvement_data: Optional[str]       # Cải tiến từ RL Agent
    trading_strategy: Optional[str]       # Chiến lược giao dịch
    risk_assessment: Optional[str]        # Đánh giá rủi ro
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Hàm tiện ích ---
def save_to_vectorstore(assessment: str, symbol: str) -> str:
    """Lưu đánh giá rủi ro vào vector store."""
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "type": "risk_assessment",
        "timeframe": "3mo"
    }
    content = f"Risk Assessment (3mo):\n{assessment}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved risk assessment to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save risk assessment to vector store: {e}"

# --- Các Node của Graph ---

def generate_risk_assessment(state: RiskManagementState) -> RiskManagementState:
    """Node: Đánh giá rủi ro và đề xuất chiến lược giảm thiểu."""
    market_data = state["market_data"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not market_data or not improvement_data or not trading_strategy:
        return {
            **state,
            "risk_assessment": "Không thể đánh giá rủi ro do thiếu dữ liệu.",
            "messages": messages + [AIMessage(content="Không thể đánh giá rủi ro do thiếu dữ liệu.")]
        }

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    # Chuẩn bị dữ liệu thị trường để đưa vào prompt
    market_summary = (
        f"- Giá đóng cửa: {format_val(market_data.get('latest_close'))}\n"
        f"- Khối lượng giao dịch: {format_val(market_data.get('latest_volume'))}\n"
        f"- SMA Khối lượng (20 ngày): {format_val(market_data.get('volume_sma_20'))}\n"
        f"- Biến động (hàng năm): {format_val(market_data.get('volatility', None) * 100)}%\n"
        f"- Thay đổi giá (%): { {k: f'{format_val(v)}%' for k, v in market_data.get('price_changes', {}).items()} }"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là chuyên gia quản lý rủi ro chứng khoán.

                1. Đánh giá rủi ro cho mã chứng khoán dựa trên:
                - Dữ liệu thị trường (biến động, thanh khoản, thay đổi giá).
                - Cải tiến quyết định từ RL Agent.
                - Chiến lược giao dịch hiện tại.

                2. Định dạng đầu ra:
                - Đánh giá các yếu tố rủi ro (biến động, thanh khoản, chiến lược).
                - Đề xuất chiến lược giảm thiểu rủi ro (dừng lỗ, phân bổ vốn, phòng ngừa).
                - Giải thích lý do chọn các chiến lược.
                - Tối đa 250 từ, rõ ràng, cô đọng."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Đánh giá rủi ro và đề xuất chiến lược giảm thiểu rủi ro cho mã chứng khoán {symbol}.
                **Dữ liệu Thị trường ({market_date}):**
                {market_summary}

                **Cải tiến Quyết định:**
                {improvement_data}

                **Chiến lược Giao dịch:**
                {trading_strategy}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_date": market_data.get('latest_date', 'N/A'),
        "market_summary": market_summary,
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
    }
    chain = prompt_template | llm

    assessment_message = chain.invoke(prompt_input)
    assessment_content = assessment_message.content
    save_to_vectorstore(assessment_content, symbol)

    return {
        **state,
        "risk_assessment": assessment_content,
        "messages": messages + [AIMessage(content=f"Đánh giá rủi ro và chiến lược giảm thiểu cho {symbol}:\n{assessment_content}")]
    }

def format_final_output(state: RiskManagementState) -> RiskManagementState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    risk_assessment = state["risk_assessment"]
    market_data = state["market_data"]
    improvement_data = state["improvement_data"]
    trading_strategy = state["trading_strategy"]
    messages = state["messages"]

    if not risk_assessment or not market_data:
        output = f"Không thể đánh giá rủi ro cho {symbol} do thiếu dữ liệu."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}


    output = f"""# Quản lý Rủi ro {symbol}
            **Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            ## Tóm tắt Thị trường
            {market_data}
            
            ## Gợi ý Cải tiến Quyết định
            {improvement_data}

            ## Chiến lược Giao dịch
            {trading_strategy}

            ## Đánh giá Rủi ro và Chiến lược Giảm thiểu
            {risk_assessment}
            """

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---

def build_risk_management_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(RiskManagementState)

    workflow.add_node("generate_risk_assessment", generate_risk_assessment)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "generate_risk_assessment")
    workflow.add_edge("generate_risk_assessment", "format_final_output")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def risk_management_agent(symbol: str, market_data: Optional[str], improvement_data: Optional[str], trading_strategy: Optional[str]):
    """Chạy agent quản lý rủi ro."""
    initial_state = {
        "messages": [HumanMessage(content=f"Đánh giá rủi ro và đề xuất chiến lược giảm thiểu cho {symbol}.")],
        "symbol": symbol,
        "market_data": market_data,
        "improvement_data": improvement_data,
        "trading_strategy": trading_strategy,
        "risk_assessment": None,
        "final_output": None,
    }

    graph = build_risk_management_workflow()
    result = graph.invoke(initial_state)
    
    return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    
    # Dữ liệu mẫu để thử nghiệm
    market_context = (
        f"Bối cảnh Thị trường cho {symbol_to_analyze}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: Công nghệ đang có dấu hiệu tích lũy.\n"
        f"- Tin tức: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    low_level_data = (
        f"Cải tiến Quyết định cho {symbol_to_analyze}:\n"
        f"- Tránh bán khi RSI gần overbought nhưng MACD vẫn bullish.\n"
        f"- Tín hiệu kháng cự tại 155 không đáng tin cậy trong xu hướng tăng mạnh.\n"
        f"- Đề xuất: Tăng tỷ trọng khi giá phá vỡ kháng cự với khối lượng cao."
    )
    high_level_data = (
        f"Chiến lược Giao dịch cho {symbol_to_analyze}:\n"
        f"- Chiến lược: Momentum Trading\n"
        f"- Triển khai: Mua khi giá phá vỡ kháng cự 155 với khối lượng cao, bán khi MACD chuyển bearish.\n"
        f"- Lý do: Xu hướng tăng mạnh, MACD bullish, khối lượng giao dịch tăng."
    )

    print(f"\n--- Bắt đầu Workflow Quản lý Rủi ro cho {symbol_to_analyze} ---")
    result = risk_management_agent(
        symbol_to_analyze,
        market_context,
        low_level_data,
        high_level_data
    )
    print(f"\n--- Workflow Quản lý Rủi ro cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Quản lý Rủi ro Cuối cùng ---")
    print(result)