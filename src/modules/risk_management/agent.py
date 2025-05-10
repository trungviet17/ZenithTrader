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

# --- Định nghĩa State cho Risk Management Agent ---
class RiskManagementState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    symbol: str
    market_context: Optional[str]
    low_level_data: Optional[str]
    trading_strategy: Optional[str]
    risk_assessment: Optional[str]
    final_output: Optional[str]

# --- Các Node của Graph ---

def generate_risk_assessment(state: RiskManagementState) -> RiskManagementState:
    """Node: Đánh giá rủi ro và đề xuất chiến lược giảm thiểu."""
    market_context = state["market_context"] or "Không có dữ liệu thị trường."
    low_level_data = state["low_level_data"] or "Không có dữ liệu phân tích kỹ thuật."
    trading_strategy = state["trading_strategy"] or "Không có chiến lược giao dịch."
    symbol = state["symbol"]
    messages = state["messages"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là chuyên gia quản lý rủi ro chứng khoán.

                1. Đánh giá rủi ro cho mã chứng khoán dựa trên:
                - Dữ liệu thị trường (biến động, thanh khoản, thay đổi giá).
                - Dữ liệu phân tích kỹ thuật.
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
                **Thông tin Thị trường:**
                {market_context}
                
                **Dữ liệu Phân tích Kỹ thuật:**
                {low_level_data}

                **Chiến lược Giao dịch:**
                {trading_strategy}"""
            ),
        ]
    )

    prompt_input = {
        "messages": messages,
        "symbol": symbol,
        "market_context": market_context,
        "low_level_data": low_level_data,
        "trading_strategy": trading_strategy,
    }
    chain = prompt_template | llm

    try:
        assessment_message = chain.invoke(prompt_input)
        assessment_content = assessment_message.content
    except Exception as e:
        assessment_content = f"Lỗi khi đánh giá rủi ro: {str(e)}"

    return {
        **state,
        "risk_assessment": assessment_content,
        "messages": messages + [AIMessage(content=f"Đánh giá rủi ro và chiến lược giảm thiểu cho {symbol}:\n{assessment_content}")]
    }

def format_final_output(state: RiskManagementState) -> RiskManagementState:
    """Node: Định dạng kết quả cuối cùng."""
    symbol = state["symbol"]
    risk_assessment = state["risk_assessment"] or "Không có đánh giá rủi ro."
    messages = state["messages"]

    output = f"""# Quản lý Rủi ro {symbol}
            ## Đánh giá Rủi ro và Chiến lược Giảm thiểu
            {risk_assessment}
            """

    return {
        **state,
        "final_output": output,
        "messages": messages + [AIMessage(content=output)]
    }

# --- Xây dựng Graph ---
def build_workflow():
    """Xây dựng và biên dịch graph LangGraph."""
    workflow = StateGraph(RiskManagementState)

    workflow.add_node("generate_risk_assessment", generate_risk_assessment)
    workflow.add_node("format_final_output", format_final_output)

    workflow.add_edge(START, "generate_risk_assessment")
    workflow.add_edge("generate_risk_assessment", "format_final_output")
    workflow.add_edge("format_final_output", END)

    return workflow.compile()

def risk_management_agent(symbol: str, market_context: Optional[str] = None, 
                        low_level_data: Optional[str] = None, 
                        trading_strategy: Optional[str] = None) -> str:
    """Chạy agent quản lý rủi ro."""
    initial_state = RiskManagementState(
        messages=[HumanMessage(content=f"Đánh giá rủi ro và đề xuất chiến lược giảm thiểu cho {symbol}.")],
        symbol=symbol,
        market_context=market_context,
        low_level_data=low_level_data,
        trading_strategy=trading_strategy,
        risk_assessment=None,
        final_output=None,
    )

    try:
        graph = build_workflow()
        result = graph.invoke(initial_state)
        return result.get("final_output", "Không có kết quả cuối cùng hoặc đã xảy ra lỗi.")
    except Exception as e:
        return f"Lỗi khi chạy agent: {str(e)}"

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
    low_level_data = (
        f"Dữ liệu Phân tích Kỹ thuật cho {symbol_to_analyze}:\n"
        f"- Giá hiện tại: 150 USD.\n"
        f"- Khối lượng giao dịch: 1 triệu cổ phiếu.\n"
        f"- Chỉ báo RSI: 65 (hơi quá mua).\n"
        f"- Đường trung bình động 50 ngày: 145 USD.\n"
        f"- Đường trung bình động 200 ngày: 140 USD."
    )
    trading_strategy = (
        f"Chiến lược Giao dịch cho {symbol_to_analyze}:\n"
        f"- Mua vào khi giá phá vỡ 150 với khối lượng lớn.\n"
        f"- Dừng lỗ tại 145 và chốt lời tại 160.\n"
        f"- Theo dõi các chỉ báo kỹ thuật để điều chỉnh chiến lược."
    )

    print(f"\n--- Bắt đầu Workflow Quản lý Rủi ro cho {symbol_to_analyze} ---")
    result = risk_management_agent(
        symbol_to_analyze,
        market_context,
        low_level_data,
        trading_strategy
    )
    print(f"\n--- Workflow Quản lý Rủi ro cho {symbol_to_analyze} Hoàn thành ---")
    print("\n--- Kết quả Quản lý Rủi ro Cuối cùng ---")
    print(result)