import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, List, Optional
from uuid import uuid4
from datetime import datetime
import pytz

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Khởi tạo LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# --- Định nghĩa State cho Strategy Agent ---
class StrategyState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_content: Optional[str]           # Thông tin thị trường
    low_level_data: Optional[str]          # Suy luận từ Research Module
    high_level_data: Optional[str]       # Cải tiến từ RL Agent
    trading_strategy: Optional[str]       # Chiến lược giao dịch được chọn
    final_output: Optional[str]           # Kết quả cuối cùng

# --- Các Node của Graph ---

def generate_trading_strategy(state: StrategyState) -> StrategyState:
    """Node: Tạo chiến lược giao dịch dựa trên dữ liệu đầu vào."""
    market_content = state["market_content"]
    low_level_data = state["low_level_data"]
    high_level_data = state["high_level_data"]
    symbol = state["symbol"]
    messages = state["messages"]

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
                **Thông tin Thị trường:**
                {market_summary}

                **Phân tích Kỹ thuật:**
                {low_level_data}

                **Cải tiến Quyết định:**
                {high_level_data}

                {available_strategies}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_summary": market_content,
        "low_level_data": low_level_data,
        "high_level_data": high_level_data,
        "available_strategies": available_strategies,
    }
    chain = prompt_template | llm

    strategy_message = chain.invoke(prompt_input)
    strategy_content = strategy_message.content

    return {
        **state,
        "trading_strategy": strategy_content,
        "messages": messages + [AIMessage(content=f"Chiến lược giao dịch đề xuất cho {symbol}:\n{strategy_content}")]
    }

def format_final_output(state: StrategyState) -> StrategyState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    trading_strategy = state["trading_strategy"]
    messages = state["messages"]

    output = f"""# Chiến lược Giao dịch {symbol}:

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

def strategy_agent(symbol: str, market_content: Optional[str], low_level_data: Optional[str], high_level_data: Optional[str]):
    """Chạy agent chiến lược giao dịch."""
    initial_state = {
        "messages": [HumanMessage(content=f"Đề xuất chiến lược giao dịch cho {symbol}.")],
        "symbol": symbol,
        "market_content": market_content,
        "low_level_data": low_level_data,
        "high_level_data": high_level_data,
        "trading_strategy": None,
        "final_output": None,
    }
    graph = build_strategy_workflow()
    result = graph.invoke(initial_state)
    
    return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")

graph = build_strategy_workflow()

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    symbol_to_analyze = "AAPL"
    
    market_context = (
        f"Bối cảnh Thị trường cho {symbol_to_analyze}:\n"
        f"- Tổng quan: Thị trường chung có xu hướng đi ngang trong vài tuần qua.\n"
        f"- Ngành: Công nghệ đang có dấu hiệu tích lũy.\n"
        f"- Tin tức: Không có tin tức trọng yếu nào gần đây ảnh hưởng đến giá."
    )
    low_level_data = (
        f"Dữ liệu Phân tích Kỹ thuật cho {symbol_to_analyze}:\n"
        f"- Giá hiện tại: 150 USD.\n"
        f"- Khối lượng giao dịch: 1 triệu cổ phiếu.\n"
        f"- Chỉ báo RSI: 65 (hơi quá mua).\n"
        f"- Đường trung bình động 50 ngày: 145 USD.\n"
        f"- Đường trung bình động 200 ngày: 140 USD."
    )
    high_level_data = (
        f"Cải tiến Quyết định cho {symbol_to_analyze}:\n"
        f"- Tránh bán khi RSI gần overbought nhưng MACD vẫn bullish.\n"
        f"- Tín hiệu kháng cự tại 155 không đáng tin cậy trong xu hướng tăng mạnh.\n"
        f"- Đề xuất: Tăng tỷ trọng khi giá phá vỡ kháng cự với khối lượng cao."
    )

    print(f"\n--- Bắt đầu Workflow Chiến lược Giao dịch cho {symbol_to_analyze} ---")
    result = strategy_agent(
        symbol_to_analyze,
        market_content=market_context,
        low_level_data=low_level_data,
        high_level_data=high_level_data
    )
    print(f"\n--- Workflow Chiến lược Giao dịch cho {symbol_to_analyze} Hoàn thành ---")

    print("\n--- Kết quả Chiến lược Cuối cùng ---")
    print(result)