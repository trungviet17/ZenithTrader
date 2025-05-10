import yfinance as yf
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from uuid import uuid4
import json

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
from dotenv import load_dotenv
import os 
load_dotenv()


google_api_key = os.getenv("GEMINI_API_KEY")




# --- Khởi tạo LLM, Embeddings và Vector Store ---
# llm = ChatOllama(model="cogito:3b")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key,
)

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
def calculate_technical_indicators(symbol: str, interval: str = "1d", period: str = "3mo") -> Dict:
    """
    Lấy dữ liệu giá cổ phiếu từ Yahoo Finance và tính toán các chỉ báo kỹ thuật.

    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        interval: Khoảng cách giữa các điểm dữ liệu (vd: 1h, 1d, 5d, 1wk).
        period: Khoảng thời gian lấy dữ liệu (vd: 1mo, 3mo, 6mo, 1y, ytd, max).

    Returns:
        Dictionary chứa các chỉ báo kỹ thuật đã tính toán hoặc thông báo lỗi.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return {"error": f"Không tìm thấy dữ liệu cho {symbol} với khoảng thời gian={period}, interval={interval}"}

        windows = [5, 10, 20, 50, 100, 200]
        for w in windows:
            if len(df) >= w:
                df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
            else:
                df[f'MA{w}'] = np.nan

        bb_window = 20
        if len(df) >= bb_window:
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
            df['BB_Std'] = df['Close'].rolling(window=bb_window).std()
            df['BBU'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BBL'] = df['BB_Middle'] - 2 * df['BB_Std']
        else:
            df['BB_Middle'], df['BBU'], df['BBL'] = np.nan, np.nan, np.nan

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

        span1, span2, signal_span = 12, 26, 9
        if len(df) >= span2:
            ema_12 = df['Close'].ewm(span=span1, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=span2, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=signal_span, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        else:
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = np.nan, np.nan, np.nan

        vol_sma_window = 20
        if len(df) >= vol_sma_window:
            df['Vol_SMA20'] = df['Volume'].rolling(window=vol_sma_window).mean()
        else:
            df['Vol_SMA20'] = np.nan

        latest = df.iloc[-1]
        indicator_keys = [f"MA{w}" for w in windows] + ["BBU", "BB_Middle", "BBL Common Indicators ", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "Vol_SMA20"]
        indicators = {k: (round(v, 2) if pd.notna(v) and isinstance(v, (int, float)) else None) for k, v in indicators.items()}

        signals = {}
        close_price = latest.get('Close')
        if close_price is not None:
            if indicators.get("MA20") and indicators.get("MA50"):
                signals["Trend_ShortMid"] = "Tăng giá (MA20 > MA50)" if indicators["MA20"] > indicators["MA50"] else "Giảm giá (MA20 < MA50)"
            if indicators.get("MA50") and indicators.get("MA200"):
                signals["Trend_MidLong"] = "Tăng giá (MA50 > MA200)" if indicators["MA50"] > indicators["MA200"] else "Giảm giá (MA50 < MA200)"
            if indicators.get("RSI"):
                signals["RSI_Signal"] = "Quá mua" if indicators["RSI"] > 70 else "Quá bán" if indicators["RSI"] < 30 else "Trung tính"
            if indicators.get("MACD") is not None and indicators.get("MACD_Signal") is not None:
                signals["MACD_Signal"] = (
                    "Giao cắt tăng giá / Đà tăng" if indicators["MACD"] > indicators["MACD_Signal"] and indicators.get("MACD_Hist", 0) > 0 else
                    "Giao cắt giảm giá / Đà giảm" if indicators["MACD"] < indicators["MACD_Signal"] and indicators.get("MACD_Hist", 0) < 0 else
                    "Trung tính / Yếu dần"
                )
            if indicators.get("BBU") and indicators.get("BBL"):
                signals["BB_Signal"] = (
                    "Giá vượt dải trên (có thể quá mua/breakout)" if close_price > indicators["BBU"] else
                    "Giá dưới dải dưới (có thể quá bán/breakdown)" if close_price < indicators["BBL"] else
                    "Giá trong dải Bollinger"
                )

        changes = {}
        if close_price is not None:
            for days in [5, 20, 60]:
                if len(df) > days:
                    start_price = df['Close'].iloc[-days-1]
                    if start_price != 0:
                        changes[f"{days}d_change_pct"] = round(((close_price - start_price) / start_price) * 100, 2)
                    else:
                        changes[f"{days}d_change_pct"] = None
                else:
                    changes[f"{days}d_change_pct"] = None

        return json.loads(json.dumps({
            "symbol": symbol,
            "latest_close": round(close_price, 2) if close_price is not None else None,
            "latest_date": df.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None,
            "indicators": indicators,
            "signals": signals if signals else {"info": "Không đủ dữ liệu để tạo tín hiệu"},
            "price_changes_pct": changes
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi tính toán chỉ báo cho {symbol}: {str(e)}"}

@tool
def analyze_patterns_trends(symbol: str, interval: str = "1d", period: str = "6mo") -> Dict:
    """
    Phân tích mẫu giá, xu hướng, hỗ trợ và kháng cự dựa trên dữ liệu giá cổ phiếu từ Yahoo Finance.

    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        interval: Khoảng cách giữa các điểm dữ liệu.
        period: Khoảng thời gian lấy dữ liệu.

    Returns:
        Dictionary chứa mẫu giá, xu hướng, và hỗ trợ/kháng cự hoặc thông báo lỗi.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return {"error": f"Không tìm thấy dữ liệu cho {symbol} với khoảng thời gian={period}, interval={interval}"}

        patterns = []
        if len(df) >= 5:
            recent_df = df.tail(5)
            for i in range(len(recent_df)):
                row = recent_df.iloc[i]
                body_size = abs(row['Open'] - row['Close'])
                range_size = row['High'] - row['Low']
                if range_size > 0 and body_size / range_size < 0.1:
                    patterns.append(f"Khả năng có nến Doji vào {recent_df.index[i].strftime('%Y-%m-%d')}")

        if len(df) >= 60:
            rolling_max = df['High'].rolling(window=30).max()
            rolling_min = df['Low'].rolling(window=30).min()
            if df['Close'].iloc[-1] < rolling_max.iloc[-2] * 0.98 and df['High'].iloc[-30:-1].max() > rolling_max.iloc[-2] * 0.99:
                patterns.append("Cảnh báo khả năng có mẫu hình Double Top (cần xác nhận thêm)")
            if df['Close'].iloc[-1] > rolling_min.iloc[-2] * 1.02 and df['Low'].iloc[-30:-1].min() < rolling_min.iloc[-2] * 1.01:
                patterns.append("Cảnh báo khả năng có mẫu hình Double Bottom (cần xác nhận thêm)")

        trends = {}
        if len(df) >= 50:
            ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            trends["Xu_huong_ngan_han"] = "Tăng giá (MA20 trên MA50)" if pd.notna(ma20) and pd.notna(ma50) and ma20 > ma50 else "Giảm giá (MA20 dưới MA50)"
        else:
            trends["Xu_huong_ngan_han"] = "Không đủ dữ liệu"

        if len(df) >= 200:
            ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
            trends["Xu_huong_dai_han"] = "Tăng giá (MA50 trên MA200)" if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200 else "Giảm giá (MA50 dưới MA200)"
        else:
            trends["Xu_huong_dai_han"] = "Không đủ dữ liệu"

        support_resistance = {}
        if len(df) >= 30:
            recent_data_sr = df.tail(30)
            highest_high = recent_data_sr['High'].max()
            lowest_low = recent_data_sr['Low'].min()
            last_close = recent_data_sr['Close'].iloc[-1]

            support_resistance["Ho_tro_gan_nhat_30d"] = round(lowest_low, 2) if pd.notna(lowest_low) else None
            support_resistance["Khang_cu_gan_nhat_30d"] = round(highest_high, 2) if pd.notna(highest_high) else None

            if pd.notna(highest_high) and pd.notna(lowest_low) and highest_high > lowest_low:
                diff = highest_high - lowest_low
                support_resistance["Fib_0.236_retracement"] = round(highest_high - 0.236 * diff, 2)
                support_resistance["Fib_0.382_retracement"] = round(highest_high - 0.382 * diff, 2)
                support_resistance["Fib_0.500_retracement"] = round(highest_high - 0.500 * diff, 2)
                support_resistance["Fib_0.618_retracement"] = round(highest_high - 0.618 * diff, 2)

        return json.loads(json.dumps({
            "symbol": symbol,
            "patterns_detected": patterns if patterns else ["Không có mẫu hình rõ ràng được phát hiện tự động"],
            "current_trends": trends,
            "support_resistance_levels": support_resistance
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi phân tích mẫu hình và xu hướng cho {symbol}: {str(e)}"}

@tool
def analyze_volume_profile(symbol: str, interval: str = "1d", period: str = "3mo") -> Dict:
    """
    Phân tích hồ sơ khối lượng giao dịch (volume profile) để xác định các vùng giá quan trọng.

    Args:
        symbol: Mã cổ phiếu (ticker symbol).
        interval: Khoảng cách giữa các điểm dữ liệu.
        period: Khoảng thời gian lấy dữ liệu.

    Returns:
        Dictionary chứa thông tin về vùng giá có khối lượng cao/thấp và vùng giá trị (value area).
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return {"error": f"Không tìm thấy dữ liệu cho {symbol} với khoảng thời gian={period}, interval={interval}"}

        # Phân tích hồ sơ khối lượng
        price_bins = np.histogram(df['Close'], bins=50, weights=df['Volume'])[0]
        price_levels = np.histogram(df['Close'], bins=50)[1]

        # Tìm vùng giá có khối lượng cao (Point of Control - POC)
        poc_index = np.argmax(price_bins)
        poc_price = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2

        # Tính vùng giá trị (Value Area: 68% tổng khối lượng)
        total_volume = df['Volume'].sum()
        value_area_volume = total_volume * 0.68
        sorted_indices = np.argsort(price_bins)[::-1]
        cumulative_volume = 0
        value_area_bins = []
        for idx in sorted_indices:
            cumulative_volume += price_bins[idx]
            value_area_bins.append(idx)
            if cumulative_volume >= value_area_volume:
                break

        value_area_high = (price_levels[max(value_area_bins) + 1] + price_levels[max(value_area_bins)]) / 2
        value_area_low = (price_levels[min(value_area_bins) + 1] + price_levels[min(value_area_bins)]) / 2

        return json.loads(json.dumps({
            "symbol": symbol,
            "point_of_control": round(poc_price, 2),
            "value_area_high": round(value_area_high, 2),
            "value_area_low": round(value_area_low, 2),
            "latest_date": df.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi phân tích hồ sơ khối lượng cho {symbol}: {str(e)}"}

@tool
def fetch_market_sentiment(symbol: str) -> Dict:
    """
    Lấy thông tin tâm lý thị trường (ví dụ: VIX) để bổ sung bối cảnh phân tích.

    Args:
        symbol: Mã cổ phiếu (ticker symbol).

    Returns:
        Dictionary chứa thông tin tâm lý thị trường hoặc thông báo lỗi.
    """
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period="1mo", interval="1d")

        if vix_data.empty:
            return {"error": "Không tìm thấy dữ liệu VIX"}

        latest_vix = vix_data['Close'].iloc[-1]
        vix_signal = (
            "Tâm lý thị trường biến động cao (sợ hãi)" if latest_vix > 30 else
            "Tâm lý thị trường ổn định (tự tin)" if latest_vix < 20 else
            "Tâm lý thị trường trung tính"
        )

        return json.loads(json.dumps({
            "symbol": symbol,
            "vix_level": round(latest_vix, 2),
            "vix_signal": vix_signal,
            "latest_date": vix_data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not vix_data.empty else None
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi lấy dữ liệu tâm lý thị trường: {str(e)}"}

tools = [calculate_technical_indicators, analyze_patterns_trends, analyze_volume_profile, fetch_market_sentiment]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

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
                """Bạn là một chuyên gia phân tích kỹ thuật thị trường tài chính. Nhiệm vụ của bạn là cung cấp phân tích kỹ thuật chuyên sâu cho mã chứng khoán dựa trên bối cảnh thị trường.

                Yêu cầu:
                1. Phân tích các chỉ báo kỹ thuật chính (MA, RSI, MACD, Bollinger Bands, Volume Profile).
                2. Nhận diện mẫu giá (Doji, Double Top/Bottom), xu hướng ngắn hạn và dài hạn.
                3. Xác định các vùng hỗ trợ, kháng cự, vùng giá trị (value area) và điểm kiểm soát (POC).
                4. Đánh giá tâm lý thị trường (dựa trên VIX hoặc các yếu tố khác).
                5. Đưa ra dự báo ngắn hạn (1-2 tuần) rõ ràng, dựa trên các tín hiệu kỹ thuật.
                6. Định dạng đầu ra:
                   - Tóm tắt chỉ báo kỹ thuật.
                   - Mẫu giá và xu hướng.
                   - Hỗ trợ, kháng cự, vùng giá trị.
                   - Tâm lý thị trường.
                   - Dự báo ngắn hạn.
                   - Tối đa 300 từ, ngắn gọn, mạch lạc.
                7. Sử dụng các công cụ:
                   - calculate_technical_indicators
                   - analyze_patterns_trends
                   - analyze_volume_profile
                   - fetch_market_sentiment"""
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
    else:
        analysis_content = analysis_message.content
        prompt_critique = f"""
            Đánh giá phân tích kỹ thuật cho {symbol}:
            - Phân tích: {analysis_content}
            - Bối cảnh thị trường: {market_context}
            Yêu cầu:
            1. Đánh giá tính chính xác, đầy đủ và logic của phân tích.
            2. Xác định 1-2 điểm mạnh và 1-2 điểm yếu.
            Đầu ra: Tối đa 100 từ, ngắn gọn, rõ ràng.
            """
        critique_message = llm.invoke(prompt_critique)
        critique_content = critique_message.content

        prompt_query = f"""
            Tạo câu hỏi truy vấn tìm kiếm lịch sử phân tích kỹ thuật cho {symbol} dựa trên:
            - Phân tích: {analysis_content}
            - Đánh giá: {critique_content}
            Yêu cầu: Truy vấn cụ thể, tập trung vào các chỉ báo (RSI, MACD, MA), mẫu giá, và vùng giá trị chính.
            Đầu ra: Tối đa 50 từ.
            """
        query_message = llm.invoke(prompt_query)
        query_content = query_message.content

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
    query = state["query"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]

    prompt_analysis = f"""
        Tinh chỉnh phân tích kỹ thuật cho {symbol} dựa trên:
        - Phân tích ban đầu: {response}
        - Đánh giá: {critique}
        - Lịch sử: {reflection_data}
        Yêu cầu:
        1. Cải thiện phân tích theo đánh giá và lịch sử.
        2. Đảm bảo tính chính xác, đầy đủ, và phù hợp với bối cảnh thị trường.
        3. Giữ định dạng: tóm tắt chỉ báo, mẫu giá, hỗ trợ/kháng cự, tâm lý, dự báo.
        Đầu ra: Tối đa 300 từ.
        """
    refined_analysis_message = llm.invoke(prompt_analysis)
    refined_analysis_content = refined_analysis_message.content

    prompt_critique = f"""
        Đánh giá phân tích kỹ thuật đã tinh chỉnh cho {symbol}:
        - Phân tích: {refined_analysis_content}
        Yêu cầu:
        1. Đánh giá tính chính xác, đầy đủ, và logic.
        2. Xác định 1-2 điểm mạnh và 1-2 điểm yếu.
        Đầu ra: Tối đa 100 từ.
        """
    refined_critique_message = llm.invoke(prompt_critique)
    refined_critique_content = refined_critique_message.content

    prompt_query = f"""
        Tạo câu hỏi truy vấn tìm kiếm lịch sử cho phân tích kỹ thuật đã tinh chỉnh của {symbol}:
        - Phân tích: {refined_analysis_content}
        - Đánh giá: {refined_critique_content}
        Yêu cầu: Truy vấn cụ thể, tập trung vào các chỉ báo, mẫu giá, và vùng giá trị.
        Đầu ra: Tối đa 50 từ.
        """
    refined_query_message = llm.invoke(prompt_query)
    refined_query_content = refined_query_message.content

    prompt_references = f"""
        Tạo danh sách tài liệu tham khảo được sử dụng cho phân tích kỹ thuật đã tinh chỉnh của {symbol}:
        - Phân tích: {refined_analysis_content}
        - Lịch sử: {reflection_data}
        Yêu cầu: Liệt kê các nguồn dữ liệu được sử dụng từ lịch sử theo mẫu:
        - [1]: [Mô tả ngắn gọn về nội dung]
        - Đảm bảo tính chính xác và liên quan đến phân tích.
        Đầu ra: Tối đa 50 từ.
        """
    refined_references_message = llm.invoke(prompt_references)
    refined_references_content = refined_references_message.content

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

    if not final_analysis:
        output = f"Không thể hoàn thành phân tích (3 tháng) cho {symbol} do thiếu dữ liệu."
        return {**state, "final_output": output, "messages": messages + [AIMessage(content=output)]}

    save_to_vectorstore(final_analysis, symbol, market_context)

    output = f"""
        # Phân tích Kỹ thuật {symbol}
        **Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Bối cảnh Thị trường:**  
        {market_context}

        ## Phân tích Chi tiết
        {final_analysis}
        """

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