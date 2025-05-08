import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import pytz

from langchain_ollama import ChatOllama, OllamaEmbeddings
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
embeddings = OllamaEmbeddings(model="cogito:3b")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./strategy_agent/qdrant_data"
COLLECTION_NAME = "strategy_history"
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

# --- Định nghĩa State cho Strategy Agent ---
class StrategyState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_data: Optional[Dict]           # Thông tin thị trường
    research_data: Optional[str]          # Suy luận từ Research Module
    improvement_data: Optional[str]       # Cải tiến từ RL Agent
    trading_strategy: Optional[str]       # Chiến lược giao dịch được chọn
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Hàm tiện ích ---
def save_to_vectorstore(strategy: str, symbol: str) -> str:
    """Lưu chiến lược giao dịch vào vector store."""
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "type": "trading_strategy",
        "timeframe": "3mo"
    }
    content = f"Trading Strategy (3mo):\n{strategy}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved trading strategy to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save trading strategy to vector store: {e}"

# --- Các Node của Graph ---

def generate_trading_strategy(state: StrategyState) -> StrategyState:
    """Node: Tạo chiến lược giao dịch dựa trên dữ liệu đầu vào."""
    market_data = state["market_data"]
    research_data = state["research_data"]
    improvement_data = state["improvement_data"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not market_data or not research_data or not improvement_data:
        return {
            **state,
            "trading_strategy": "Không thể tạo chiến lược do thiếu dữ liệu.",
            "messages": messages + [AIMessage(content="Không thể tạo chiến lược do thiếu dữ liệu.")]
        }

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    # Chuẩn bị dữ liệu thị trường để đưa vào prompt
    market_summary = (
        f"- Giá đóng cửa: {format_val(market_data.get('latest_close'))}\n"
        f"- Khối lượng giao dịch: {format_val(market_data.get('latest_volume'))}\n"
        f"- SMA Khối lượng (20 ngày): {format_val(market_data.get('volume_sma_20'))}\n"
        f"- Thay đổi giá (%): { {k: f'{format_val(v)}%' for k, v in market_data.get('price_changes', {}).items()} }"
    )

    # Định nghĩa các chiến lược có sẵn
    available_strategies = """
**Chiến lược Có sẵn:**
1. **Trend Following**: Mua khi giá phá vỡ kháng cự với khối lượng cao, bán khi giá thủng hỗ trợ.
2. **Mean Reversion**: Mua khi giá chạm hỗ trợ mạnh hoặc RSI oversold, bán khi giá chạm kháng cự hoặc RSI overbought.
3. **Warren Buffett (Value Investing)**: Mua cổ phiếu có giá trị nội tại cao, giữ dài hạn khi thị trường ổn định.
4. **Momentum Trading**: Mua khi xu hướng tăng mạnh (MACD bullish, giá trên MA20), bán khi momentum suy yếu.
"""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là chuyên gia chiến lược giao dịch chứng khoán.

                1. Đề xuất chiến lược giao dịch cho mã chứng khoán dựa trên:
                - Dữ liệu thị trường (giá, khối lượng, thay đổi giá).
                - Phân tích kỹ thuật từ Research Module.
                - Cải tiến quyết định từ RL Agent.

                2. Định dạng đầu ra:
                - Phân tích trạng thái thị trường hiện tại.
                - Chọn chiến lược phù hợp từ các chiến lược có sẵn hoặc đề xuất chiến lược tùy chỉnh.
                - Giải thích lý do chọn chiến lược và cách triển khai (mua/bán/giữ, thời điểm, điều kiện).
                - Tối đa 250 từ, rõ ràng, cô đọng."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Đề xuất chiến lược giao dịch cho mã chứng khoán {symbol}.
                **Dữ liệu Thị trường ({market_date}):**
                {market_summary}

                **Phân tích Kỹ thuật:**
                {research_data}

                **Cải tiến Quyết định:**
                {improvement_data}

                {available_strategies}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_date": market_data.get('latest_date', 'N/A'),
        "market_summary": market_summary,
        "research_data": research_data,
        "improvement_data": improvement_data,
        "available_strategies": available_strategies,
    }
    chain = prompt_template | llm

    strategy_message = chain.invoke(prompt_input)
    strategy_content = strategy_message.content
    save_to_vectorstore(strategy_content, symbol)

    return {
        **state,
        "trading_strategy": strategy_content,
        "messages": messages + [AIMessage(content=f"Chiến lược giao dịch đề xuất cho {symbol}:\n{strategy_content}")]
    }

def format_final_output(state: StrategyState) -> StrategyState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    trading_strategy = state["trading_strategy"]
    market_data = state["market_data"]
    research_data = state["research_data"]
    improvement_data = state["improvement_data"]
    messages = state["messages"]

    if not trading_strategy or not market_data:
        output = f"Không thể tạo chiến lược giao dịch cho {symbol} do thiếu dữ liệu."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    output = f"""# Chiến lược Giao dịch {symbol}
**Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dữ liệu đến ngày:** {market_data.get('latest_date', 'N/A')}
**Giá đóng cửa cuối cùng:** {format_val(market_data.get('latest_close'))}

## Tóm tắt Dữ liệu Thị trường
- **Khối lượng giao dịch gần nhất:** {format_val(market_data.get('latest_volume'))}
- **SMA Khối lượng (20 ngày):** {format_val(market_data.get('volume_sma_20'))}
- **Thay đổi giá (20 ngày):** {format_val(market_data.get('price_changes', {}).get('20d_change_pct'))}%
- **Thay đổi giá (60 ngày):** {format_val(market_data.get('price_changes', {}).get('60d_change_pct'))}%

## Phân tích Kỹ thuật
{research_data}

## Cải tiến Quyết định
{improvement_data}

## Chiến lược Giao dịch Đề xuất
{trading_strategy}
"""

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---

def build_strategy_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(StrategyState)

    workflow.add_node("generate_trading_strategy", generate_trading_strategy)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "generate_trading_strategy")
    workflow.add_edge("generate_trading_strategy", "format_final_output")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def strategy_agent(symbol: str, market_data: Optional[Dict], research_data: Optional[str], improvement_data: Optional[str]):
    """Chạy agent chiến lược giao dịch."""
    initial_state = {
        "messages": [HumanMessage(content=f"Đề xuất chiến lược giao dịch cho {symbol}.")],
        "symbol": symbol,
        "market_data": market_data,
        "research_data": research_data,
        "improvement_data": improvement_data,
        "trading_strategy": None,
        "final_output": None,
    }
    graph = build_strategy_workflow()
    result = graph.invoke(initial_state)
    
    return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    
    # Dữ liệu mẫu để thử nghiệm
    sample_market_data = {
        "symbol": symbol_to_analyze,
        "latest_close": 150.0,
        "latest_volume": 1000000,
        "latest_date": "2025-05-08",
        "volume_sma_20": 950000,
        "price_changes": {
            "5d_change_pct": 2.5,
            "10d_change_pct": 5.0,
            "20d_change_pct": 10.0,
            "60d_change_pct": 15.0
        }
    }
    sample_research_data = (
        f"Phân tích Kỹ thuật {symbol_to_analyze} (3 tháng):\n"
        f"- Xu hướng: Bullish (MA20 > MA50).\n"
        f"- RSI: 65 (Neutral, gần Overbought).\n"
        f"- MACD: Bullish Crossover.\n"
        f"- Hỗ trợ: 145; Kháng cự: 155.\n"
        f"- Dự báo ngắn hạn: Tăng nhẹ trong vài tuần tới."
    )
    sample_improvement_data = (
        f"Cải tiến Quyết định cho {symbol_to_analyze}:\n"
        f"- Tránh bán khi RSI gần overbought nhưng MACD vẫn bullish.\n"
        f"- Tín hiệu kháng cự tại 155 không đáng tin cậy trong xu hướng tăng mạnh.\n"
        f"- Đề xuất: Tăng tỷ trọng khi giá phá vỡ kháng cự với khối lượng cao."
    )

    print(f"\n--- Bắt đầu Workflow Chiến lược Giao dịch cho {symbol_to_analyze} ---")
    result = strategy_agent(
        symbol_to_analyze,
        sample_market_data,
        sample_research_data,
        sample_improvement_data
    )
    print(f"\n--- Workflow Chiến lược Giao dịch cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Chiến lược Cuối cùng ---")
    print(result)