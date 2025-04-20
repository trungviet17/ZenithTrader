import pandas as pd
from datetime import datetime
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import pytz

# Initialize LLM and vector store
llm = OllamaLLM(model="cogito:3b")
embeddings = OllamaEmbeddings(model="cogito:3b")
client = QdrantClient(path="./qdrant_data")

# client.create_collection(
#     collection_name="low_level_reflection",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="low_level_reflection",
    embedding=embeddings,
)
# Add sample documents
content_1 = "LangChain is a framework for developing applications powered by language models."
content_2 = "It provides a standard interface for LLMs and tools to build applications."
document_1 = Document(page_content=content_1, metadata={"timestamp": "2023-10-01"})
document_2 = Document(page_content=content_2, metadata={"timestamp": "2023-10-02"})
documents = [document_1, document_2]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

@tool
def process_trading_decisions(trades: list) -> str:
    """
    Xử lý dữ liệu giao dịch, tính toán lợi nhuận tích lũy và tạo biểu đồ.
    """
    try:
        if not trades:
            return "Error: No trading decisions provided."

        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        if 'timestamp' not in df.columns or 'symbol' not in df.columns or 'decision' not in df.columns or 'price' not in df.columns:
            return "Error: Invalid trade data format."

        # Calculate returns
        df['return'] = df.groupby('symbol')['price'].pct_change().fillna(0)
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1

        # Calculate statistics
        correct_decisions = df[df['return'] > 0].shape[0]
        total_decisions = df.shape[0]
        success_rate = (correct_decisions / total_decisions) * 100 if total_decisions > 0 else 0

        return "\n".join([
            f"Tổng số giao dịch: {total_decisions}",
            f"Tỷ lệ giao dịch thành công: {success_rate:.2f}%",
            "Chi tiết lợi nhuận theo mã:",
            *[
                f"- {symbol}: Lợi nhuận trung bình {df[df['symbol'] == symbol]['return'].mean() * 100:.2f}%"
                for symbol in df['symbol'].unique()
            ]
        ])
    except Exception as e:
        return f"Error: Failed to process trading decisions: {str(e)}"

# Define state
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    trades: List[Dict[str, Any]]
    trade_analysis: str
    llm_reflection: str
    paste_lessons: str
    lessons_learned: str
    output: str

# Define nodes
def fetch_trade_data(state: AgentState) -> AgentState:
    """Node để xử lý dữ liệu giao dịch."""
    trades = state["trades"]
    result = process_trading_decisions.invoke({"trades": trades})
    return {"trade_analysis": result}

def analyze_decisions(state: AgentState) -> AgentState:
    """Node để phân tích quyết định giao dịch."""
    trade_analysis = state["trade_analysis"]
    prompt = (
        f"Dựa trên dữ liệu giao dịch:\n{trade_analysis}\n"
        "Phân tích các quyết định giao dịch, xác định điểm mạnh, điểm yếu, "
        "và đề xuất cải tiến. Trả về tối đa 300 token."
    )
    response = llm.invoke(prompt)
    return {"llm_reflection": response}

def generate_lessons(state: AgentState) -> AgentState:
    """Node để rút ra bài học."""
    llm_reflection = state["llm_reflection"]
    prompt = (
        f"Dựa trên phân tích:\n{llm_reflection}\n"
        "Rút ra các bài học chính từ các quyết định giao dịch, "
        "tập trung vào những thành công và sai lầm. "
        "Trả về tối đa 200 token."
    )
    lessons = llm.invoke(prompt)
    
    # Store lessons in vector store
    analysis_doc = Document(
        id=f"lessons_{datetime.now(pytz.UTC).isoformat()}",
        page_content=lessons,
        metadata={"timestamp": datetime.now(pytz.UTC).isoformat()}
    )
    vector_store.add_documents(documents=[analysis_doc])
    
    return {"lessons_learned": lessons}

def paste_lessons(state: AgentState) -> AgentState:
    """Node để dán bài học."""
    trade_analysis = state["trade_analysis"]
    prompt = (
        f"Dựa trên phân tích:\n{trade_analysis}\n"
        "Tạo một truy vấn cho LLM cho các bài học quá khứ liên quan.\n"
        "Truy vấn này sẽ được sử dụng để lấy thông tin bổ sung từ LLM.\n"
        "Đầu ra chỉ cần là một câu hỏi, không cần giải thích hay thông tin bổ sung.\n"
    )
    query = llm.invoke(prompt)
    paste_lessons = retriever.invoke(query)
    return {"paste_lessons": paste_lessons}

def format_output(state: AgentState) -> AgentState:
    """Node để định dạng output."""
    output = (
        f"Phân tích giao dịch:\n{state['trade_analysis']}\n"
        f"Phân tích quyết định:\n{state['llm_reflection']}\n"
        f"Bài học rút ra:\n{state['paste_lessons']}\n"
    )
    return {"output": output}

# Create workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("fetch_trade_data", fetch_trade_data)
workflow.add_node("analyze_decisions", analyze_decisions)
workflow.add_node("past_lessons", paste_lessons)
workflow.add_node("generate_lessons", generate_lessons)
workflow.add_node("format_output", format_output)

# Define edges
workflow.add_edge(START, "fetch_trade_data")
workflow.add_edge("fetch_trade_data", "analyze_decisions")
workflow.add_edge("fetch_trade_data", "past_lessons")
workflow.add_edge("analyze_decisions", "generate_lessons")
workflow.add_edge("analyze_decisions", "format_output")
workflow.add_edge("past_lessons", "format_output")
workflow.add_edge("format_output", END)

# Compile graph
graph = workflow.compile()

# Test
if __name__ == "__main__":
    sample_trades = [
        {"timestamp": "2023-10-01", "symbol": "AAPL", "decision": "Buy", "price": 150.0},
        {"timestamp": "2023-10-02", "symbol": "AAPL", "decision": "Sell", "price": 155.0},
        {"timestamp": "2023-10-03", "symbol": "AAPL", "decision": "Buy", "price": 300.0},
        {"timestamp": "2023-10-04", "symbol": "AAPL", "decision": "Sell", "price": 290.0}
    ]
    
    initial_state = {
        "messages": [HumanMessage(content="Phân tích các quyết định giao dịch trước đây")],
        "trades": sample_trades,
        "trade_analysis": "",
        "llm_reflection": "",
        "paste_lessons": "",
        "lessons_learned": "",
        "output": ""
    }
    
    result = graph.invoke(initial_state)
    print("Kết quả phân tích High-level Reflection:")
    print(result["output"])