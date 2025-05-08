from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from uuid import uuid4
import json

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
def evaluate_trade_performance(trade_history: List[Dict]) -> Dict:
    """
    Đánh giá hiệu suất giao dịch dựa trên lịch sử giao dịch.

    Args:
        trade_history: Danh sách các giao dịch (mỗi giao dịch chứa date, action, price, quantity, profit_loss, outcome).

    Returns:
        Dictionary chứa các chỉ số hiệu suất (tỷ lệ thắng, lợi nhuận trung bình, tỷ lệ rủi ro/lợi nhuận).
    """
    try:
        if not trade_history:
            return {"error": "Không có lịch sử giao dịch để đánh giá."}

        total_trades = len(trade_history)
        wins = sum(1 for trade in trade_history if trade["outcome"] == "Profit")
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        avg_profit_loss = np.mean([trade["profit_loss"] for trade in trade_history]) if total_trades > 0 else 0
        total_profit = sum(trade["profit_loss"] for trade in trade_history)

        # Tính tỷ lệ rủi ro/lợi nhuận (giả sử rủi ro là giá vào * 2% mỗi giao dịch)
        risk_reward_ratios = []
        for trade in trade_history:
            entry_price = trade["price"]
            risk = entry_price * 0.02  # Giả định rủi ro 2%
            reward = trade["profit_loss"] / trade["quantity"] if trade["quantity"] > 0 else 0
            if risk > 0:
                risk_reward_ratios.append(reward / risk)
        avg_risk_reward = np.mean(risk_reward_ratios) if risk_reward_ratios else 0

        return json.loads(json.dumps({
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "avg_profit_loss": round(avg_profit_loss, 2),
            "total_profit": round(total_profit, 2),
            "avg_risk_reward_ratio": round(avg_risk_reward, 2)
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi đánh giá hiệu suất giao dịch: {str(e)}"}

@tool
def simulate_alternative_trades(trade_history: List[Dict], technical_analysis: str) -> Dict:
    """
    Mô phỏng các kịch bản giao dịch thay thế để tối ưu hóa lợi nhuận.

    Args:
        trade_history: Danh sách các giao dịch.
        technical_analysis: Phân tích kỹ thuật cung cấp thông tin về hỗ trợ, kháng cự, tín hiệu.

    Returns:
        Dictionary chứa các đề xuất giao dịch thay thế và lợi nhuận tiềm năng.
    """
    try:
        if not trade_history or not technical_analysis:
            return {"error": "Thiếu lịch sử giao dịch hoặc phân tích kỹ thuật."}

        # Giả lập các kịch bản thay thế dựa trên hỗ trợ/kháng cự từ phân tích kỹ thuật
        support = None
        resistance = None
        for line in technical_analysis.split("\n"):
            if "Hỗ trợ" in line:
                support = float(line.split(":")[-1].strip().split(";")[0])
            if "Kháng cự" in line:
                resistance = float(line.split(":")[-1].strip().split(";")[0])

        alternative_trades = []
        for trade in trade_history:
            original_price = trade["price"]
            action = trade["action"]
            quantity = trade["quantity"]
            profit_loss = trade["profit_loss"]

            if action == "Buy" and support and original_price > support:
                alt_price = support  # Mua tại mức hỗ trợ
                alt_profit = (original_price - alt_price) * quantity
                alternative_trades.append({
                    "original_trade": trade,
                    "alternative_action": "Buy",
                    "alternative_price": alt_price,
                    "potential_profit_improvement": round(alt_profit, 2)
                })
            elif action == "Sell" and resistance and original_price < resistance:
                alt_price = resistance  # Bán tại mức kháng cự
                alt_profit = (alt_price - original_price) * quantity
                alternative_trades.append({
                    "original_trade": trade,
                    "alternative_action": "Sell",
                    "alternative_price": alt_price,
                    "potential_profit_improvement": round(alt_profit, 2)
                })

        return json.loads(json.dumps({
            "alternative_trades": alternative_trades if alternative_trades else ["Không có kịch bản thay thế khả thi."]
        }, ignore_nan=True))
    except Exception as e:
        return {"error": f"Lỗi khi mô phỏng giao dịch thay thế: {str(e)}"}

tools = [evaluate_trade_performance, simulate_alternative_trades]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto")

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

    trade_summary = "\n".join([
        f"- Trade {t['date']} ({t['action']} @ {t['price']}): {t['reason']} Outcome: {t['outcome']} (P/L: {t['profit_loss']}). Analysis: {t['analysis']}"
        for t in trade_history
    ]) if trade_history else "Không có lịch sử giao dịch."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là một chuyên gia phân tích giao dịch chứng khoán, có khả năng đánh giá và tối ưu hóa các quyết định giao dịch.

                Yêu cầu:
                1. Phân tích các giao dịch trong quá khứ của mã chứng khoán dựa trên:
                   - Bối cảnh thị trường.
                   - Phân tích kỹ thuật (hỗ trợ, kháng cự, RSI, MACD, v.v.).
                   - Lịch sử giao dịch (ngày, hành động, giá, lý do, kết quả).
                2. Đánh giá từng giao dịch:
                   - Quyết định đúng/sai? Tại sao (dựa trên tín hiệu kỹ thuật, thị trường)?
                   - Nếu sai, đề xuất cải thiện (thay đổi điểm vào/ra, khối lượng, thời điểm).
                3. Tối ưu hóa lợi nhuận:
                   - Mô phỏng các kịch bản giao dịch thay thế để tăng lợi nhuận.
                4. Tóm tắt bài học kinh nghiệm:
                   - Từ thành công/sai lầm, rút ra bài học áp dụng cho bối cảnh hiện tại.
                5. Đưa ra khuyến nghị giao dịch hiện tại dựa trên phân tích kỹ thuật và thị trường.
                6. Định dạng đầu ra:
                   - Tóm tắt bối cảnh thị trường và phân tích kỹ thuật.
                   - Đánh giá từng giao dịch (đúng/sai, lý do, cải thiện).
                   - Kịch bản thay thế để tối ưu lợi nhuận.
                   - Bài học kinh nghiệm.
                   - Khuyến nghị giao dịch hiện tại.
                   - Tối đa 400 từ, rõ ràng, mạch lạc.
                7. Sử dụng các công cụ:
                   - evaluate_trade_performance
                   - simulate_alternative_trades"""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                """Phân tích quyết định giao dịch cho mã chứng khoán {symbol}.
                Bối cảnh thị trường: {market_data}
                Phân tích kỹ thuật: {technical_analysis}
                Lịch sử giao dịch: {trade_summary}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_data": market_data,
        "technical_analysis": technical_analysis,
        "trade_summary": trade_summary,
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
            Đánh giá phân tích quyết định giao dịch cho {symbol}:
            - Phân tích: {analysis_content}
            - Bối cảnh thị trường: {market_data}
            Yêu cầu:
            1. Đánh giá tính chính xác, logic và thực tiễn của phân tích.
            2. Xác định 1-2 điểm mạnh và 1-2 điểm yếu.
            Đầu ra: Tối đa 100 từ, ngắn gọn, rõ ràng.
            """
        critique_message = llm.invoke(prompt_critique)
        critique_content = critique_message.content

        prompt_query = f"""
            Tạo câu hỏi truy vấn tìm kiếm lịch sử phân tích giao dịch cho {symbol} dựa trên:
            - Phân tích: {analysis_content}
            - Đánh giá: {critique_content}
            Yêu cầu: Truy vấn cụ thể, tập trung vào tín hiệu kỹ thuật, quyết định giao dịch, và bài học kinh nghiệm.
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
    query = state["query"]
    symbol = state["symbol"]
    messages = state["messages"]
    iteration = state["reflection_iteration"]

    prompt_analysis = f"""
        Tinh chỉnh phân tích quyết định giao dịch cho {symbol} dựa trên:
        - Phân tích ban đầu: {response}
        - Đánh giá: {critique}
        - Lịch sử: {reflection_data}
        Yêu cầu:
        1. Cải thiện theo đánh giá và lịch sử (thêm kịch bản thay thế, phân tích sâu hơn).
        2. Đảm bảo tính chính xác, thực tiễn và phù hợp với bối cảnh thị trường.
        3. Tóm tắt bài học kinh nghiệm và khuyến nghị giao dịch hiện tại.
        Đầu ra: Tối đa 400 từ.
        """
    refined_analysis_message = llm.invoke(prompt_analysis)
    refined_analysis_content = refined_analysis_message.content

    prompt_critique = f"""
        Đánh giá phân tích quyết định giao dịch đã tinh chỉnh cho {symbol}:
        - Phân tích: {refined_analysis_content}
        Yêu cầu:
        1. Đánh giá tính chính xác, logic và thực tiễn.
        2. Xác định 1-2 điểm mạnh và 1-2 điểm yếu..
        Đầu ra: Tối đa 100 từ.
        """
    refined_critique_message = llm.invoke(prompt_critique)
    refined_critique_content = refined_critique_message.content

    prompt_query = f"""
        Tạo câu hỏi truy vấn tìm kiếm lịch sử cho phân tích quyết định giao dịch đã tinh chỉnh của {symbol}:
        - Phân tích: {refined_analysis_content}
        - Đánh giá: {refined_critique_content}
        Yêu cầu: Truy vấn cụ thể, tập trung vào tín hiệu kỹ thuật, quyết định giao dịch, và bài học.
        Đầu ra: Tối đa 50 từ.
        """
    refined_query_message = llm.invoke(prompt_query)
    refined_query_content = refined_query_message.content

    return {
        **state,
        "response": refined_analysis_content,
        "critique": refined_critique_content,
        "query": refined_query_content,
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
        **Ngày phân tích:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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

def high_level_reflection_agent(symbol: str, market_data: Optional[str], technical_analysis: Optional[str], trade_history: Optional[List[Dict]], max_reflections: int = 2):
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

    result = high_level_reflection_agent(
        symbol_to_analyze,
        sample_market_data,
        sample_technical_analysis,
        sample_trade_history,
        max_reflections
    )
    print(result)