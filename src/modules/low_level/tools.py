import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pytz

# Initialize vector store and retriever
embeddings = OllamaEmbeddings(model="cogito:3b")
vector_store = InMemoryVectorStore(embedding=embeddings)
# Add sample documents
content_1 = "LangChain is a framework for developing applications powered by language models."
content_2 = "It provides a standard interface for LLMs and tools to build applications."
document_1 = Document(id="1", page_content=content_1, metadata={"baz": "bar"})
document_2 = Document(id="2", page_content=content_2, metadata={"bar": "baz"})
documents = [document_1, document_2]
vector_store.add_documents(documents=documents)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Công cụ lấy dữ liệu K-line và nhận xét
@tool
def get_kline_data(symbol: str, period: str = "3mo", interval: str = "1d") -> str:
    """
    Lấy dữ liệu K-line từ Yahoo Finance, tính toán giá thay đổi, chỉ báo kỹ thuật (MA5, MA20, Bollinger Bands),
    và trả về nhận xét dạng văn bản cho các khung thời gian: ngắn hạn (10 ngày), trung hạn (30 ngày), dài hạn (60 ngày).
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return f"Error: No data found for {symbol} in the given period."

        # Tính chỉ báo kỹ thuật
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['BB_Middle'] = df['MA20']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BBU'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BBL'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Xác định xu hướng giá
        df['Trend'] = 'Neutral'
        df.loc[df['MA5'] > df['MA20'], 'Trend'] = 'Bullish'
        df.loc[df['MA5'] < df['MA20'], 'Trend'] = 'Bearish'

        # Lọc dữ liệu cho các khung thời gian
        ny_tz = pytz.timezone('America/New_York')
        today = datetime.now(ny_tz)
        short_term_start = today - timedelta(days=10)
        medium_term_start = today - timedelta(days=30)
        long_term_start = today - timedelta(days=60)

        short_term_data = df[df.index >= short_term_start]
        medium_term_data = df[df.index >= medium_term_start]
        long_term_data = df[df.index >= long_term_start]

        # Tính toán giá thay đổi (%)
        def calc_price_change(data):
            if len(data) < 2:
                return 0.0
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            return ((end_price - start_price) / start_price) * 100

        short_term_change = calc_price_change(short_term_data)
        medium_term_change = calc_price_change(medium_term_data)
        long_term_change = calc_price_change(long_term_data)

        # Nhận xét
        remarks = []
        if not short_term_data.empty:
            latest_close = short_term_data['Close'].iloc[-1]
            trend = short_term_data['Trend'].iloc[-1]
            ma5_ma20 = "trên" if short_term_data['MA5'].iloc[-1] > short_term_data['MA20'].iloc[-1] else "dưới"
            bb_status = ("quá mua" if latest_close > short_term_data['BBU'].iloc[-1] else 
                        "quá bán" if latest_close < short_term_data['BBL'].iloc[-1] else "ổn định")
            remarks.append(
                f"Ngắn hạn (10 ngày qua): Giá thay đổi {short_term_change:.2f}%. "
                f"MA5 {ma5_ma20} MA20, xu hướng {trend.lower()}. "
                f"Giá {bb_status} theo Bollinger Bands."
            )
        else:
            remarks.append("Ngắn hạn: Không đủ dữ liệu.")

        if not medium_term_data.empty:
            latest_close = medium_term_data['Close'].iloc[-1]
            trend = medium_term_data['Trend'].iloc[-1]
            ma5_ma20 = "trên" if medium_term_data['MA5'].iloc[-1] > medium_term_data['MA20'].iloc[-1] else "dưới"
            bb_status = ("quá mua" if latest_close > medium_term_data['BBU'].iloc[-1] else 
                        "quá bán" if latest_close < medium_term_data['BBL'].iloc[-1] else "ổn định")
            remarks.append(
                f"Trung hạn (30 ngày qua): Giá thay đổi {medium_term_change:.2f}%. "
                f"MA5 {ma5_ma20} MA20, xu hướng {trend.lower()}. "
                f"Giá {bb_status} theo Bollinger Bands."
            )
        else:
            remarks.append("Trung hạn: Không đủ dữ liệu.")

        if not long_term_data.empty:
            latest_close = long_term_data['Close'].iloc[-1]
            trend = long_term_data['Trend'].iloc[-1]
            ma5_ma20 = "trên" if long_term_data['MA5'].iloc[-1] > long_term_data['MA20'].iloc[-1] else "dưới"
            bb_status = ("quá mua" if latest_close > long_term_data['BBU'].iloc[-1] else 
                        "quá bán" if latest_close < long_term_data['BBL'].iloc[-1] else "ổn định")
            remarks.append(
                f"Dài hạn (60 ngày qua): Giá thay đổi {long_term_change:.2f}%. "
                f"MA5 {ma5_ma20} MA20, xu hướng {trend.lower()}. "
                f"Giá {bb_status} theo Bollinger Bands."
            )
        else:
            remarks.append("Dài hạn: Không đủ dữ liệu.")

        return "\n".join([
            f"Dữ liệu K-line cho {symbol}:",
            f"- Giá đóng cửa gần nhất: {df['Close'].iloc[-1]:.2f}",
            f"- MA5 gần nhất: {df['MA5'].iloc[-1]:.2f}",
            f"- MA20 gần nhất: {df['MA20'].iloc[-1]:.2f}",
            f"- BBU gần nhất: {df['BBU'].iloc[-1]:.2f}",
            f"- BBL gần nhất: {df['BBL'].iloc[-1]:.2f}",
            "Nhận xét:",
            *remarks
        ])
    except Exception as e:
        return f"Error: Failed to fetch data for {symbol}: {str(e)}"

# Công cụ giả lập thông tin thị trường
@tool
def get_market_intelligence(symbol: str) -> str:
    """
    Giả lập thông tin thị trường dạng văn bản.
    """
    return (
        f"Thông tin thị trường cho {symbol}:\n"
        f"- Tin tức mới nhất: Báo cáo quý gần nhất vượt kỳ vọng, dự báo tăng trưởng doanh thu 5%.\n"
        f"- Xu hướng ngành: Ngành công nghệ tăng trưởng nhờ nhu cầu cao.\n"
        f"- Sự kiện gần đây: Ra mắt sản phẩm mới tuần trước."
    )

# Định nghĩa trạng thái
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    symbol: str
    exchange: str
    industry: str
    description: str
    kline_data: str
    market_intel: str
    llm_reasoning: str
    query: str
    output: str

# Khởi tạo LLM
llm = OllamaLLM(model="cogito:3b")

# Định nghĩa các node
def fetch_kline_data(state: AgentState) -> AgentState:
    """Node để lấy dữ liệu K-line."""
    symbol = state["symbol"]
    result = get_kline_data.invoke({"symbol": symbol})
    return {"kline_data": result}

def fetch_market_intelligence(state: AgentState) -> AgentState:
    """Node để lấy thông tin thị trường."""
    symbol = state["symbol"]
    result = get_market_intelligence.invoke({"symbol": symbol})
    return {"market_intel": result}

def analyze_trends(state: AgentState) -> AgentState:
    """Node để phân tích xu hướng giá cho tất cả khung thời gian."""
    kline_data = state["kline_data"]
    market_intel = state["market_intel"]
    prompt = (
        f"Dựa trên dữ liệu K-line:\n{kline_data}\n"
        f"và thông tin thị trường:\n{market_intel}\n"
        "Phân tích xu hướng giá trong các khoảng thời gian:\n"
        "- Ngắn hạn (10 ngày qua, dự đoán 5 ngày tới).\n"
        "- Trung hạn (30 ngày qua, dự đoán 15 ngày tới).\n"
        "- Dài hạn (60 ngày qua, dự đoán 30 ngày tới).\n"
        "Xác định xu hướng (bullish/bearish/neutral), đánh giá Bollinger Bands (quá mua/quá bán/ổn định), "
        "kết hợp tin tức để giải thích và dự đoán xu hướng tương lai. Trả về tối đa 300 token."
    )
    response = llm.invoke(prompt)
    return {"llm_reasoning": response}

def analyze_past(state: AgentState) -> AgentState:
    """Node để tạo truy vấn về lịch sử giá và lấy thông tin từ vector store."""
    llm_reasoning = state["llm_reasoning"]
    prompt = (
        f"Dựa trên phân tích:\n{llm_reasoning}\n"
        "Tạo một truy vấn cho LLM để hỏi về lịch sử giá liên quan đến phân tích hiện tại.\n"
        "Truy vấn này sẽ được sử dụng để lấy thông tin bổ sung từ LLM.\n"
        "Đầu ra chỉ cần là một câu hỏi, không cần giải thích hay thông tin bổ sung.\n"
    )
    query = llm.invoke(prompt)
    retrieved = retriever.invoke(query)
    # Extract page_content from the first document, or use a default message if empty
    retrieved_content = retrieved[0].page_content if retrieved else "Không tìm thấy thông tin liên quan."
    return {"query": retrieved_content}

def format_output(state: AgentState) -> AgentState:
    """Node để định dạng output."""
    output = (
        f"Phân tích K-line:\n{state['kline_data']}\n"
        f"Thông tin thị trường:\n{state['market_intel']}\n"
        f"Phân tích xu hướng:\n{state['llm_reasoning']}\n"
        f"Truy vấn cho LLM:\n{state['query']}\n"
    )
    return {"output": output}

# Tạo workflow
workflow = StateGraph(AgentState)

# Thêm các node
workflow.add_node("fetch_kline_data", fetch_kline_data)
workflow.add_node("fetch_market_intelligence", fetch_market_intelligence)
workflow.add_node("llm_analyze", analyze_trends)
workflow.add_node("generate_query", analyze_past)
workflow.add_node("format_output", format_output)

# Định nghĩa luồng
workflow.add_edge(START, "fetch_kline_data")
workflow.add_edge("fetch_kline_data", "fetch_market_intelligence")
workflow.add_edge("fetch_market_intelligence", "llm_analyze")
workflow.add_edge("llm_analyze", "generate_query")
workflow.add_edge("generate_query", "format_output")
workflow.add_edge("format_output", END)

# Biên dịch graph
graph = workflow.compile()

# Test
if __name__ == "__main__":
    symbol = "AAPL"
    exchange = "NASDAQ"
    industry = "Technology"
    description = "Apple Inc. designs, manufactures, and markets consumer electronics, software, and services."
    
    initial_state = {
        "messages": [HumanMessage(content=f"Phân tích biến động giá của {symbol}")],
        "symbol": symbol,
        "exchange": exchange,
        "industry": industry,
        "description": description,
        "kline_data": "",
        "market_intel": "",
        "llm_reasoning": "",
        "query": "",
        "output": ""
    }
    
    result = graph.invoke(initial_state)
    print("Kết quả phân tích Low-level Reflection:")
    print(result["output"])