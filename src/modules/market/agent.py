import yfinance as yf
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from uuid import uuid4

from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
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
QDRANT_PATH = "./market_intelligent/qdrant_data"
COLLECTION_NAME = "market_intelligent_history"
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

# --- Định nghĩa State cho Market Intelligent Agent ---
class MarketIntelligentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_summary: Optional[str]         # Tóm tắt thị trường hiện tại (last market)
    history_query: Optional[str]          # Truy vấn để tìm thông tin lịch sử
    past_market: Optional[str]            # Thông tin thị trường lịch sử
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Tools cho Market Intelligent Agent ---

@tool
def fetch_market_data(symbol: str, interval: str = "1d") -> Dict:
    """
    Lấy dữ liệu thị trường từ Yahoo Finance cho 3 tháng gần nhất.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo", interval=interval)
        if df.empty:
            return {"error": f"No data found for {symbol} with period=3mo, interval={interval}"}
        
        ohlcv_data = {
            "dates": df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "open": df['Open'].tolist(),
            "high": df['High'].tolist(),
            "low": df['Low'].tolist(),
            "close": df['Close'].tolist(),
            "volume": df['Volume'].tolist(),
        }
        
        changes = {}
        for days in [5, 10, 20, 60]:
            if len(df) > days:
                start_price = df['Close'].iloc[-days-1]
                end_price = df['Close'].iloc[-1]
                if start_price != 0:
                    changes[f"{days}d_change_pct"] = ((end_price - start_price) / start_price) * 100
                else:
                    changes[f"{days}d_change_pct"] = None
            else:
                changes[f"{days}d_change_pct"] = None

        volume_sma = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None

        return {
            "symbol": symbol,
            "period_fetched": "3mo",
            "interval": interval,
            "stock_df": df.to_dict(),
            "ohlcv_data": ohlcv_data,
            "latest_close": float(df['Close'].iloc[-1]),
            "latest_volume": int(df['Volume'].iloc[-1]),
            "latest_date": df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "price_changes": changes,
            "volume_sma_20": float(volume_sma) if volume_sma is not None else None
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

@tool
def fetch_market_news(symbol: str) -> str:
    """
    Lấy tin tức thị trường liên quan đến cổ phiếu (mô phỏng).
    """
    return (
        f"Tin tức Thị trường cho {symbol}:\n"
        f"- {symbol} vừa công bố báo cáo tài chính quý gần nhất với doanh thu vượt kỳ vọng.\n"
        f"- Ngành công nghệ đang nhận được sự chú ý do xu hướng AI.\n"
        f"- Không có tin tức tiêu cực đáng kể trong 3 tháng qua."
    )

@tool
def fetch_market_context(symbol: str) -> str:
    """
    Lấy bối cảnh thị trường chung liên quan đến cổ phiếu (mô phỏng).
    """
    return (
        f"Bối cảnh Thị trường cho {symbol}:\n"
        f"- Thị trường chứng khoán toàn cầu đang trong xu hướng tăng nhẹ nhờ kỳ vọng lãi suất ổn định.\n"
        f"- Ngành công nghệ, đặc biệt là AI, đang dẫn đầu về dòng tiền đầu tư.\n"
        f"- Cổ phiếu {symbol} có tương quan cao với chỉ số Nasdaq."
    )

tools = [fetch_market_data, fetch_market_news, fetch_market_context]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

# --- Hàm tiện ích ---
def save_to_vectorstore(summary: str, symbol: str) -> str:
    """Lưu tóm tắt thị trường vào vector store."""
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(),
        "symbol": symbol,
        "type": "market_summary",
        "timeframe": "3mo"
    }
    content = f"Market Summary (3mo):\n{summary}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved market summary to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save market summary to vector store: {e}"

# --- Các Node của Graph ---

def generate_market_summary(state: MarketIntelligentState) -> MarketIntelligentState:
    """Node: Tạo tóm tắt thị trường hiện tại (last market)."""
    symbol = state["symbol"]
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là chuyên gia phân tích thị trường chứng khoán.

                1. Tạo tóm tắt thị trường hiện tại (last market) cho mã chứng khoán {symbol} dựa trên:
                   - Dữ liệu thị trường (giá, khối lượng, thay đổi giá).
                   - Tin tức thị trường.
                   - Bối cảnh thị trường chung.

                2. Hành động:
                   - Sử dụng các công cụ 'fetch_market_data', 'fetch_market_news', và 'fetch_market_context' để lấy thông tin cần thiết.
                   - Đảm bảo dữ liệu được định dạng rõ ràng trước khi tạo tóm tắt.

                3. Định dạng đầu ra:
                   - Tóm tắt xu hướng giá và khối lượng giao dịch trong 3 tháng qua.
                   - Đánh giá tác động của tin tức thị trường.
                   - Phân tích ảnh hưởng của bối cảnh thị trường chung.
                   - Nhận định tổng quan về tình hình thị trường hiện tại.
                   - Tối đa 250 từ, rõ ràng, cô đọng."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Tạo tóm tắt thị trường hiện tại (last market) cho mã chứng khoán {symbol}."""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
    }
    chain = prompt_template | llm_with_tools

    response = chain.invoke(prompt_input)

    # Kiểm tra nếu response yêu cầu gọi tool
    if hasattr(response, 'tool_calls') and response.tool_calls:
        return {
            **state,
            "messages": messages + [response]
        }

    # Nếu không có tool call, giả định response là tóm tắt
    summary_content = response.content
    save_to_vectorstore(summary_content, symbol)

    return {
        **state,
        "market_summary": summary_content,
        "messages": messages + [AIMessage(content=f"Tóm tắt thị trường hiện tại (Last Market) cho {symbol}:\n{summary_content}")]
    }

def generate_history_query(state: MarketIntelligentState) -> MarketIntelligentState:
    """Node: Tạo truy vấn để tìm thông tin thị trường lịch sử (past market)."""
    market_summary = state["market_summary"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not market_summary:
        return {
            **state,
            "history_query": None,
            "messages": messages + [AIMessage(content="Không thể tạo truy vấn lịch sử do thiếu tóm tắt thị trường.")]
        }

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là chuyên gia phân tích thị trường chứng khoán.

                1. Tạo truy vấn cụ thể để tìm kiếm thông tin thị trường lịch sử (past market) trong 3 tháng qua.
                2. Truy vấn nên tập trung vào:
                   - Xu hướng giá, khối lượng giao dịch, tin tức, hoặc bối cảnh thị trường tương tự hiện tại.
                   - Các giai đoạn có đặc điểm tương tự tóm tắt thị trường hiện tại.

                3. Định dạng đầu ra:
                   - Truy vấn ngắn gọn, rõ ràng.
                   - Ví dụ: "Tìm các giai đoạn trong 3 tháng qua khi {symbol} có xu hướng giá tăng và tin tức tích cực."
                   - Tối đa 50 từ."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Tạo truy vấn lịch sử cho mã chứng khoán {symbol}.
                **Tóm tắt Hiện tại (Last Market):**
                {market_summary}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_summary": market_summary,
    }
    chain = prompt_template | llm

    query_message = chain.invoke(prompt_input)
    query_content = query_message.content

    return {
        **state,
        "history_query": query_content,
        "messages": messages + [AIMessage(content=f"Truy vấn lịch sử (Past Market) cho {symbol}:\n{query_content}")]
    }

def retrieve_past_market(state: MarketIntelligentState) -> MarketIntelligentState:
    """Node: Truy vấn vector store để lấy thông tin thị trường lịch sử (past market)."""
    query = state["history_query"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not query:
        return {
            **state,
            "past_market": "Không có truy vấn lịch sử.",
            "messages": messages + [AIMessage(content="Không có truy vấn lịch sử.")]
        }

    enhanced_query = f"{query} (Xem xét trong khung thời gian 3 tháng gần đây)"
    try:
        retrieved_docs = retriever.invoke(enhanced_query)
        past_market = f"Không tìm thấy thông tin lịch sử liên quan đến: '{query}'."
        if retrieved_docs:
            past_market_list = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                past_market_list.append(
                    f"Past Market {i+1} ({metadata.get('symbol', 'N/A')}, {metadata.get('timeframe','?')}, {metadata.get('type', '?')}):\n'{content_preview}'"
                )
            past_market = f"Tìm thấy {len(retrieved_docs)} thông tin lịch sử liên quan '{query}':\n\n" + "\n\n".join(past_market_list)
    except Exception as e:
        past_market = f"Lỗi khi truy xuất lịch sử: {e}"

    return {
        **state,
        "past_market": past_market,
        "messages": messages + [AIMessage(content=f"Thông tin thị trường lịch sử (Past Market):\n{past_market}")]
    }

def format_final_output(state: MarketIntelligentState) -> MarketIntelligentState:
    """Node: Định dạng kết quả cuối cùng, bao gồm last market và past market."""
    symbol = state["symbol"]
    market_summary = state["market_summary"]
    past_market = state["past_market"]
    messages = state["messages"]

    if not market_summary:
        output = f"Không thể hoàn thành tóm tắt thị trường cho {symbol} do thiếu tóm tắt thị trường."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    output = f"""# Tóm tắt Thị trường {symbol}
                **Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                ## Last Market (Thị trường Hiện tại)
                ### Phân tích Tổng quan
                {market_summary}

                ## Past Market (Thị trường Lịch sử)
                {past_market}
                """

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

# --- Xây dựng Graph ---

def build_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(MarketIntelligentState)

    workflow.add_node("generate_market_summary", generate_market_summary)
    workflow.add_node("generate_history_query", generate_history_query)
    workflow.add_node("retrieve_past_market", retrieve_past_market)
    workflow.add_node("format_final_output", format_final_output)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "generate_market_summary")
    workflow.add_conditional_edges(
        "generate_market_summary",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "generate_history_query"
        }
    )
    workflow.add_edge("tools", "generate_market_summary")
    workflow.add_edge("generate_history_query", "retrieve_past_market")
    workflow.add_edge("retrieve_past_market", "format_final_output")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def market_intelligent_agent(symbol: str):
    """Chạy workflow Market Intelligent Agent."""
    graph = build_workflow()
    initial_state = {
        "messages": [HumanMessage(content=f"Tóm tắt thị trường cho {symbol}.")],
        "symbol": symbol,
        "market_summary": None,
        "history_query": None,
        "past_market": None,
        "final_output": None,
    }
    result = graph.invoke(initial_state)
    return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")

graph = build_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"

    print(f"\n--- Bắt đầu Workflow Tóm tắt Thị trường cho {symbol_to_analyze} ---")
    result = market_intelligent_agent(symbol_to_analyze)
    print(f"\n--- Workflow Tóm tắt Thị trường cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Tóm tắt Cuối cùng ---")
    print(result)