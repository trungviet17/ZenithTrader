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
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Khởi tạo LLM, Embeddings và Vector Store ---
llm = ChatOllama(model="cogito:3b")
embeddings = OllamaEmbeddings(model="cogito:3b")

# --- Thiết lập Qdrant ---
QDRANT_PATH = "./low_level/qdrant_data"
COLLECTION_NAME = "technical_analysis_history" # Đổi tên collection cho phù hợp
client = QdrantClient(path=QDRANT_PATH)

# Hàm tiện ích để tạo collection nếu chưa tồn tại
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
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6}) # Giảm K vì dữ liệu ít hơn

# --- Định nghĩa State cho Agent ---
class ReflectionState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    stock_data: Optional[Dict]      # Dữ liệu cổ phiếu đã xử lý (DataFrame và dict)
    indicator_data: Optional[Dict] # Dữ liệu chỉ báo kỹ thuật
    market_context: Optional[str]  # Thông tin thị trường
    current_analysis: str          # Phân tích hiện tại
    reflection_query: Optional[str]# Truy vấn reflection
    historical_insights: Optional[str] # Thông tin lịch sử
    reflection_iteration: int      # Số lần lặp Reflection
    max_reflections: int           # Số lần lặp tối đa
    final_output: Optional[str]    # Kết quả cuối cùng

# --- Tools Phân tích Kỹ thuật ---

@tool
def fetch_stock_data(symbol: str, interval: str = "1d") -> Dict:
    """
    Lấy dữ liệu giá cổ phiếu từ Yahoo Finance cho 3 tháng gần nhất.

    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        interval: Khoảng cách giữa các điểm dữ liệu (vd: 1h, 1d, 5d, 1wk). '1d' là phổ biến nhất.

    Returns:
        Dictionary chứa DataFrame, dữ liệu OHLCV dạng list hoặc thông báo lỗi.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Luôn lấy dữ liệu 3 tháng gần nhất
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

        return {
            "symbol": symbol,
            "period_fetched": "3mo", # Ghi rõ đã fetch 3 tháng
            "interval": interval,
            "stock_df": df, # Trả về DataFrame để tính toán
            "ohlcv_data": ohlcv_data,
            "latest_close": df['Close'].iloc[-1],
            "latest_date": df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

@tool
def calculate_technical_indicators(stock_data: Dict) -> Dict:
    """
    Tính toán các chỉ báo kỹ thuật phù hợp với dữ liệu 3 tháng.

    Args:
        stock_data: Dictionary chứa dữ liệu giá cổ phiếu (kết quả từ fetch_stock_data, phải có 'stock_df').

    Returns:
        Dictionary chứa các chỉ báo kỹ thuật đã tính toán hoặc thông báo lỗi.
    """
    if "error" in stock_data:
        return {"error": f"Cannot calculate indicators due to fetch error: {stock_data['error']}"}
    if "stock_df" not in stock_data or not isinstance(stock_data["stock_df"], pd.DataFrame):
         return {"error": "Missing or invalid 'stock_df' in input."}

    symbol = stock_data["symbol"]
    try:
        df = stock_data["stock_df"].copy()

        # --- Tính toán các chỉ báo phù hợp với 3 tháng ---
        windows = [5, 10, 20, 50] # Chỉ tính các MA ngắn và trung hạn
        for w in windows:
            if len(df) >= w:
                df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
            else:
                 df[f'MA{w}'] = np.nan

        # Bollinger Bands (sử dụng MA20)
        bb_window = 20
        if f'MA{bb_window}' in df.columns and len(df) >= bb_window :
            df['BB_Middle'] = df[f'MA{bb_window}']
            df['BB_Std'] = df['Close'].rolling(window=bb_window).std()
            df['BBU'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BBL'] = df['BB_Middle'] - 2 * df['BB_Std']
        else:
            df['BB_Middle'], df['BBU'], df['BBL'] = np.nan, np.nan, np.nan

        # RSI
        rsi_period = 14
        if len(df) > rsi_period:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['RSI'] = np.nan

        # MACD
        span1, span2, signal_span = 12, 26, 9
        if len(df) >= span2:
            ema_12 = df['Close'].ewm(span=span1, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=span2, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=signal_span, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        else:
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = np.nan, np.nan, np.nan

        # Volume SMA
        vol_sma_window = 20
        if len(df) >= vol_sma_window:
            df['Vol_SMA20'] = df['Volume'].rolling(window=vol_sma_window).mean()
        else:
            df['Vol_SMA20'] = np.nan

        # --- Lấy giá trị cuối cùng ---
        latest = df.iloc[-1]
        indicator_keys = ["MA5", "MA10", "MA20", "MA50", # Chỉ các MA đã tính
                          "BBU", "BB_Middle", "BBL", "RSI", "MACD",
                          "MACD_Signal", "MACD_Hist", "Vol_SMA20"]
        indicators = {k: latest.get(k, np.nan) for k in indicator_keys}
        indicators = {k: (v if pd.notna(v) else None) for k, v in indicators.items()}

        # --- Xác định tín hiệu đơn giản ---
        signals = {}
        # Trend (chỉ dựa trên MA ngắn/trung hạn)
        if indicators.get("MA20") and indicators.get("MA50"):
            if indicators["MA20"] > indicators["MA50"]: signals["Trend_ShortMid"] = "Bullish"
            elif indicators["MA20"] < indicators["MA50"]: signals["Trend_ShortMid"] = "Bearish"
            else: signals["Trend_ShortMid"] = "Neutral"
        else:
            signals["Trend_ShortMid"] = "N/A (MA50 requires ~2.5 months)" # Ghi chú nếu MA50 thiếu

        # RSI Signal
        if indicators.get("RSI"):
            if indicators["RSI"] > 70: signals["RSI_Signal"] = "Overbought"
            elif indicators["RSI"] < 30: signals["RSI_Signal"] = "Oversold"
            else: signals["RSI_Signal"] = "Neutral"
        else: signals["RSI_Signal"] = "N/A"

        # MACD Signal
        if indicators.get("MACD") is not None and indicators.get("MACD_Signal") is not None and indicators.get("MACD_Hist") is not None:
             if indicators["MACD"] > indicators["MACD_Signal"] and indicators["MACD_Hist"] > 0: signals["MACD_Signal"] = "Bullish Crossover/Momentum"
             elif indicators["MACD"] < indicators["MACD_Signal"] and indicators["MACD_Hist"] < 0: signals["MACD_Signal"] = "Bearish Crossover/Momentum"
             else: signals["MACD_Signal"] = "Neutral/Weakening"
        else: signals["MACD_Signal"] = "N/A"

        # Bollinger Bands Signal
        latest_close = latest.get('Close')
        if latest_close and indicators.get("BBU") and indicators.get("BBL"):
            if latest_close > indicators["BBU"]: signals["BB_Signal"] = "Price above Upper Band"
            elif latest_close < indicators["BBL"]: signals["BB_Signal"] = "Price below Lower Band"
            else: signals["BB_Signal"] = "Price within Bands"
        else: signals["BB_Signal"] = "N/A"

        # --- Tính toán thay đổi giá (ngắn hạn) ---
        changes = {}
        for days in [5, 10, 20, 60]: # Chỉ tính cho các khoảng thời gian trong 3 tháng
            if len(df) > days:
                start_price = df['Close'].iloc[-days-1]
                end_price = latest.get('Close')
                if start_price is not None and end_price is not None and start_price != 0:
                    changes[f"{days}d_change_pct"] = ((end_price - start_price) / start_price) * 100
                else: changes[f"{days}d_change_pct"] = None
            else: changes[f"{days}d_change_pct"] = None

        return {
            "symbol": symbol,
            "latest_close": latest.get('Close'),
            "latest_date": df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "indicators": indicators,
            "signals": signals,
            "price_changes": changes,
        }
    except Exception as e:
        return {"error": f"Failed to calculate indicators for {symbol}: {str(e)}"}

@tool
def get_market_context(symbol: str) -> str:
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

# --- Hàm tiện ích ---
def save_analysis_to_vectorstore(analysis: str, symbol: str, iteration: int, is_final: bool = False, query: Optional[str] = None, insights: Optional[str] = None) -> str:
    """Lưu phân tích, truy vấn hoặc insight vào vector store."""
    now = datetime.now(pytz.UTC)
    doc_id = str(uuid4())
    metadata = {
        "timestamp": now.isoformat(), "symbol": symbol, "iteration": iteration,
        "type": "final_analysis" if is_final else ("reflection_query_response" if query else "intermediate_analysis"),
        "timeframe": "3mo" # Thêm thông tin timeframe
    }
    content = analysis
    if query:
        metadata["reflection_query"] = query
        content = f"Reflect Query (Iter {iteration}, 3mo): {query}\nInsights:\n{insights if insights else 'N/A'}"
    elif not is_final:
        content = f"Intermediate Analysis (Iter {iteration}, 3mo):\n{analysis}"
    else:
        content = f"Final Analysis (Iter {iteration}, 3mo):\n{analysis}"

    try:
        vector_store.add_documents(documents=[Document(page_content=content, metadata=metadata)], ids=[doc_id])
        return f"Saved analysis to vector store (ID: {doc_id})"
    except Exception as e:
        return f"Failed to save analysis to vector store: {e}"

# --- Các Node của Graph ---

def get_initial_data(state: ReflectionState) -> ReflectionState:
    """Node: Lấy dữ liệu 3 tháng, tính chỉ báo và lấy bối cảnh thị trường."""
    symbol = state["symbol"]
    messages = state["messages"]

    # Lấy dữ liệu 3 tháng (mặc định interval='1d')
    fetched_data = fetch_stock_data.invoke({"symbol": symbol})
    if "error" in fetched_data:
        return {**state, "messages": messages + [AIMessage(content=f"Lỗi: Không thể lấy dữ liệu 3 tháng cho {symbol}. {fetched_data['error']}")]}

    # Tính toán chỉ báo cho 3 tháng
    indicator_data = calculate_technical_indicators.invoke({"stock_data": fetched_data})
    if "error" in indicator_data:
        return {**state, "messages": messages + [AIMessage(content=f"Lỗi: Không thể tính toán chỉ báo cho {symbol}. {indicator_data['error']}")]}

    market_context = get_market_context.invoke({"symbol": symbol})

    return {
        **state,
        "stock_data": fetched_data,
        "indicator_data": indicator_data,
        "market_context": market_context,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Đã lấy và xử lý dữ liệu 3 tháng cho {symbol}.")]
    }

def generate_initial_analysis(state: ReflectionState) -> ReflectionState:
    """Node: Tạo phân tích kỹ thuật ban đầu dựa trên dữ liệu 3 tháng."""
    indicator_data = state["indicator_data"]
    market_context = state["market_context"]
    symbol = state["symbol"]
    messages = state["messages"]

    if not indicator_data or "error" in indicator_data:
        return {**state, "current_analysis": "Không thể tạo phân tích do lỗi dữ liệu chỉ báo."}

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    # Prompt đã được cập nhật để bỏ qua các chỉ báo dài hạn
    prompt = f"""Bạn là chuyên gia phân tích kỹ thuật. Phân tích kỹ thuật chi tiết cho {symbol} dựa trên dữ liệu 3 tháng gần nhất:

            Dữ liệu Chỉ báo ({indicator_data.get('latest_date', 'N/A')}):
            - Giá đóng cửa: {format_val(indicator_data.get('latest_close'))}
            - Các đường MA: { {k: format_val(v) for k, v in indicator_data.get('indicators', {}).items() if 'MA' in k} }
            - Bollinger Bands: U={format_val(indicator_data.get('indicators', {}).get('BBU'))}, M={format_val(indicator_data.get('indicators', {}).get('BB_Middle'))}, L={format_val(indicator_data.get('indicators', {}).get('BBL'))}
            - RSI(14): {format_val(indicator_data.get('indicators', {}).get('RSI'))} ({indicator_data.get('signals', {}).get('RSI_Signal', 'N/A')})
            - MACD: MACD={format_val(indicator_data.get('indicators', {}).get('MACD'))}, Signal={format_val(indicator_data.get('indicators', {}).get('MACD_Signal'))}, Hist={format_val(indicator_data.get('indicators', {}).get('MACD_Hist'))} ({indicator_data.get('signals', {}).get('MACD_Signal', 'N/A')})
            - Tín hiệu Xu hướng (Ngắn-Trung): {indicator_data.get('signals', {}).get('Trend_ShortMid', 'N/A')}
            - Tín hiệu BBands: {indicator_data.get('signals', {}).get('BB_Signal', 'N/A')}
            - Thay đổi giá (%): { {k: f'{format_val(v)}%' for k, v in indicator_data.get('price_changes', {}).items()} }

            Bối cảnh Thị trường:
            {market_context}

            Yêu cầu (tập trung vào khung thời gian 3 tháng):
            1. Phân tích xu hướng hiện tại (ngắn hạn, trung hạn).
            2. Xác định mức hỗ trợ/kháng cự quan trọng gần đây.
            3. Đánh giá momentum (RSI, MACD).
            4. Đưa ra nhận định tổng quan và dự báo ngắn hạn (vài ngày tới vài tuần).

            Phân tích ban đầu của bạn:
            """

    analysis_message = llm.invoke(prompt)
    analysis_content = analysis_message.content
    save_analysis_to_vectorstore(analysis_content, symbol, iteration=0)

    return {
        **state,
        "current_analysis": analysis_content,
        "reflection_iteration": 0,
        "messages": messages + [AIMessage(content=f"Phân tích ban đầu (3 tháng) cho {symbol}:\n{analysis_content}")]
    }

def generate_reflection_query(state: ReflectionState) -> ReflectionState:
    """Node: Tạo câu hỏi reflection dựa trên phân tích 3 tháng."""
    current_analysis = state["current_analysis"]
    symbol = state["symbol"]
    indicator_data = state["indicator_data"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]

    if not current_analysis or iteration >= max_reflections or not indicator_data:
         return {**state, "reflection_query": None}

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    # Prompt reflection tập trung vào dữ liệu 3 tháng
    prompt = f"""Xem xét lại phân tích kỹ thuật 3 tháng cho {symbol}:
        ---
        {current_analysis}
        ---
        Chỉ báo chính:
        - Xu hướng (Ngắn-Trung): {indicator_data.get('signals', {}).get('Trend_ShortMid', 'N/A')}
        - RSI: {format_val(indicator_data.get('indicators', {}).get('RSI'))} ({indicator_data.get('signals', {}).get('RSI_Signal', 'N/A')})
        - MACD Signal: {indicator_data.get('signals', {}).get('MACD_Signal', 'N/A')}
        - Giá/BBands: {indicator_data.get('signals', {}).get('BB_Signal', 'N/A')}

        Để cải thiện phân tích, hãy đặt 1-2 câu hỏi cụ thể cần kiểm tra trong lịch sử (có thể là các phân tích 3 tháng trước đó của {symbol} hoặc mã tương tự). Tập trung vào điểm không chắc chắn hoặc tín hiệu mâu thuẫn trong khung 3 tháng.

        Ví dụ:
        - "Khi {symbol} có RSI tương tự trong 3 tháng qua, diễn biến giá thường ra sao?"
        - "Mức hỗ trợ/kháng cự nào là quan trọng nhất trong 3 tháng này?"
        - "Tín hiệu MACD {indicator_data.get('signals', {}).get('MACD_Signal', 'N/A')} gần đây có đáng tin cậy không?"

        Câu hỏi reflection của bạn:
        """

    query_message = llm.invoke(prompt)
    query_content = query_message.content

    return {
        **state,
        "reflection_query": query_content,
        "messages": messages + [AIMessage(content=f"Câu hỏi Reflection (Iter {iteration}, 3mo):\n{query_content}")]
    }

def retrieve_historical_insights(state: ReflectionState) -> ReflectionState:
    """Node: Truy vấn vector store lấy thông tin lịch sử (ưu tiên timeframe 3mo)."""
    query = state["reflection_query"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]

    if not query:
        return {**state, "historical_insights": "Không có truy vấn reflection."}

    # Bổ sung timeframe vào query để ưu tiên kết quả liên quan
    enhanced_query = f"{query} (Xem xét trong khung thời gian 3 tháng gần đây)"
    try:
        retrieved_docs = retriever.invoke(enhanced_query)
        insights = f"Không tìm thấy lịch sử liên quan đến: '{query}'." # Default message
        if retrieved_docs:
            insights_list = []
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                content_preview = doc.page_content[:200] + "..."
                insights_list.append(
                    f"Insight {i+1} ({metadata.get('symbol', 'N/A')}, {metadata.get('timeframe','?')}, {metadata.get('type', '?')}, Iter {metadata.get('iteration', '?')}):\n'{content_preview}'"
                )
            insights = f"Tìm thấy {len(retrieved_docs)} ghi chú lịch sử liên quan '{query}':\n\n" + "\n\n".join(insights_list)

        save_analysis_to_vectorstore(
            analysis="", symbol=symbol, iteration=iteration, query=query, insights=insights
        )

    except Exception as e:
        insights = f"Lỗi khi truy xuất lịch sử: {e}"

    return {
        **state,
        "historical_insights": insights,
        "messages": messages + [AIMessage(content=f"Kết quả truy vấn lịch sử:\n{insights}")]
    }

def refine_analysis(state: ReflectionState) -> ReflectionState:
    """Node: Tinh chỉnh phân tích 3 tháng dựa trên thông tin lịch sử."""
    current_analysis = state["current_analysis"]
    historical_insights = state["historical_insights"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]

    if not historical_insights or iteration >= max_reflections:
        return {**state, "reflection_iteration": iteration + 1} # Tăng iteration để dừng

    # Prompt tinh chỉnh
    prompt = f"""Cải thiện phân tích kỹ thuật 3 tháng cho {symbol}.
                Phân tích hiện tại:
                ---
                {current_analysis}
                ---
                Thông tin lịch sử để đối chiếu:
                ---
                {historical_insights}
                ---
                Yêu cầu:
                1. Xem xét thông tin lịch sử.
                2. So sánh với phân tích 3 tháng hiện tại.
                3. Cập nhật phân tích (xu hướng, hỗ trợ/kháng cự, dự báo ngắn hạn) dựa trên sự kết hợp dữ liệu hiện tại và lịch sử.
                4. Nếu lịch sử không hữu ích, ghi nhận và giữ nguyên hoặc điều chỉnh nhẹ.

                Phân tích đã tinh chỉnh (Iteration {iteration + 1}, 3mo):
                """

    refined_analysis_message = llm.invoke(prompt)
    refined_analysis_content = refined_analysis_message.content
    save_analysis_to_vectorstore(refined_analysis_content, symbol, iteration=iteration + 1)

    return {
        **state,
        "current_analysis": refined_analysis_content,
        "reflection_iteration": iteration + 1,
        "messages": messages + [AIMessage(content=f"Phân tích đã tinh chỉnh (Iter {iteration + 1}, 3mo):\n{refined_analysis_content}")]
    }

def format_final_output(state: ReflectionState) -> ReflectionState:
    """Node: Định dạng kết quả cuối cùng cho phân tích 3 tháng."""
    symbol = state["symbol"]
    final_analysis = state["current_analysis"]
    indicator_data = state["indicator_data"]
    iteration = state["reflection_iteration"]
    messages = state["messages"]

    if not indicator_data or "error" in indicator_data:
         output = f"Không thể hoàn thành phân tích (3 tháng) cho {symbol} do lỗi dữ liệu."
         return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    save_analysis_to_vectorstore(final_analysis, symbol, iteration, is_final=True)

    def format_val(value, precision=2):
        return f'{value:.{precision}f}' if value is not None else 'N/A'

    # Output đã cập nhật, bỏ chỉ báo dài hạn
    output = f"""# Phân tích Kỹ thuật {symbol}
            **Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Dữ liệu đến ngày:** {indicator_data.get('latest_date', 'N/A')}
            **Giá đóng cửa cuối cùng:** {format_val(indicator_data.get('latest_close'))}

            ## Tóm tắt Chỉ báo Kỹ thuật Chính (3 tháng)
            - **Xu hướng (Ngắn/Trung):** {indicator_data.get('signals', {}).get('Trend_ShortMid', 'N/A')}
            - **RSI(14):** {format_val(indicator_data.get('indicators', {}).get('RSI'))} ({indicator_data.get('signals', {}).get('RSI_Signal', 'N/A')})
            - **MACD Signal:** {indicator_data.get('signals', {}).get('MACD_Signal', 'N/A')}
            - **Vị trí giá/BBands:** {indicator_data.get('signals', {}).get('BB_Signal', 'N/A')}
            - **Thay đổi giá (20 ngày):** {format_val(indicator_data.get('price_changes', {}).get('20d_change_pct'))}%
            - **Thay đổi giá (60 ngày):** {format_val(indicator_data.get('price_changes', {}).get('60d_change_pct'))}%

            ## Phân tích Chi tiết

            {final_analysis}
            """

    return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}


# --- Xây dựng Graph ---

def should_continue_reflection(state: ReflectionState) -> Literal["retrieve_historical_insights", "format_final_output"]:
    """Điều kiện định tuyến: Tiếp tục reflection hay kết thúc?"""
    iteration = state["reflection_iteration"]
    max_reflections = state["max_reflections"]
    query = state["reflection_query"]

    if query and iteration < max_reflections:
        return "retrieve_historical_insights"
    else:
        return "format_final_output"

def build_reflection_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(ReflectionState)

    workflow.add_node("get_initial_data", get_initial_data)
    workflow.add_node("generate_initial_analysis", generate_initial_analysis)
    workflow.add_node("generate_reflection_query", generate_reflection_query)
    workflow.add_node("retrieve_historical_insights", retrieve_historical_insights)
    workflow.add_node("refine_analysis", refine_analysis)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "get_initial_data")
    workflow.add_edge("get_initial_data", "generate_initial_analysis")
    workflow.add_edge("generate_initial_analysis", "generate_reflection_query")
    workflow.add_conditional_edges(
        "generate_reflection_query",
        should_continue_reflection,
        {
            "retrieve_historical_insights": "retrieve_historical_insights",
            "format_final_output": "format_final_output"
        }
    )
    workflow.add_edge("retrieve_historical_insights", "refine_analysis")
    workflow.add_edge("refine_analysis", "generate_reflection_query")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()


graph = build_reflection_workflow()
# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL" # Mã cổ phiếu cần phân tích
    max_reflections = 1        # Số vòng lặp reflection (0 hoặc 1 là đủ cho 3 tháng)

    graph = build_reflection_workflow()

    initial_state = {
        "messages": [HumanMessage(content=f"Phân tích kỹ thuật {symbol_to_analyze}.")],
        "symbol": symbol_to_analyze,
        "max_reflections": max_reflections,
        "reflection_iteration": 0,
        "current_analysis": "",
        "reflection_query": None,
        "historical_insights": None,
        "stock_data": None,
        "indicator_data": None,
        "market_context": None,
        "final_output": None,
    }

    print(f"\n--- Bắt đầu Workflow Phân tích Kỹ thuật (3 tháng) cho {symbol_to_analyze} ---")
    final_result_state = graph.invoke(initial_state)
    print(f"\n--- Workflow Phân tích Kỹ thuật cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Phân tích Cuối cùng ---")
    if final_result_state.get("final_output"):
        print(final_result_state["final_output"])