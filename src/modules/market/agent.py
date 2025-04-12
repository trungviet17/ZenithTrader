from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, TypedDict
from memory.manager import MemoryManager
from market.fetchers import fetch_market_data
from market.processors import MarketProcessor
from market.summarizers import MarketSummarizer
import numpy as np

# Định nghĩa trạng thái của graph
class MarketState(TypedDict):
    symbol: str
    market_data: Dict[str, Any]
    summary: Dict[str, str]
    queries: Dict[str, str]
    query_embeddings: np.ndarray

class MarketAgent:
    """Market Intelligence Agent using LangGraph for visualization in LangGraph Dev."""

    def __init__(self):
        self.memory = MemoryManager()
        self.processor = MarketProcessor()
        self.summarizer = MarketSummarizer()
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build and configure the LangGraph workflow."""
        # Khởi tạo graph với trạng thái MarketState
        builder = StateGraph(MarketState)

        # Thêm các node vào graph
        builder.add_node("fetch_data", self._fetch_data)
        builder.add_node("process_data", self._process_data)
        builder.add_node("store_data", self._store_data)
        builder.add_node("summarize_data", self._summarize_data)
        builder.add_node("create_queries", self._create_queries)

        # Đặt entrypoint là fetch_data
        builder.add_edge(START, "fetch_data")

        # Thêm các cạnh để tạo luồng xử lý tuần tự
        builder.add_edge("fetch_data", "process_data")
        builder.add_edge("process_data", "store_data")
        builder.add_edge("store_data", "summarize_data")
        builder.add_edge("summarize_data", "create_queries")
        builder.add_edge("create_queries", END)

        # Compile graph để sử dụng
        graph = builder.compile(
            interrupt_before=[],  # Có thể thêm node để dừng và kiểm tra trạng thái trước khi chạy
            interrupt_after=[]    # Có thể thêm node để dừng và kiểm tra trạng thái sau khi chạy
        )
        
        # Đặt tên cho graph để hiển thị trong LangGraph Dev/LangSmith
        graph.name = "Market Intelligence Agent"
        
        return graph

    async def _fetch_data(self, state: MarketState) -> MarketState:
        """Fetch market data."""
        state["market_data"] = await fetch_market_data(state["symbol"])
        return state

    async def _process_data(self, state: MarketState) -> MarketState:
        """Process fetched data."""
        price_data = self.processor.process_numerical(state["market_data"]["price_data"])
        texts, news_metadata = self.processor.process_textual(state["market_data"]["news_data"])
        visual_embeddings, visual_metadata = self.processor.process_visual([])
        state["market_data"] = {
            "price_data": price_data,
            "texts": texts,
            "news_metadata": news_metadata,
            "visual_embeddings": visual_embeddings,
            "visual_metadata": visual_metadata
        }
        return state

    async def _store_data(self, state: MarketState) -> MarketState:
        """Store processed data."""
        self.memory.store_mi_numerical(state["market_data"]["price_data"])
        self.memory.store_mi_textual(
            state["market_data"]["texts"],
            state["market_data"]["news_metadata"]
        )
        self.memory.store_mi_visual(
            state["market_data"]["visual_embeddings"],
            state["market_data"]["visual_metadata"]
        )
        return state

    async def _summarize_data(self, state: MarketState) -> MarketState:
        """Summarize stored data."""
        state["summary"] = await self.summarizer.summarize(
            state["symbol"],
            state["market_data"]["price_data"],
            state["market_data"]["news_metadata"]
        )
        return state

    async def _create_queries(self, state: MarketState) -> MarketState:
        """Create diversified queries."""
        queries = {
            "short_term_impact": "Short-term price impact of recent news",
            "trend_analysis": "Bullish or bearish trend based on latest data",
            "historical_event": "Similar past events affecting price"
        }
        query_embeddings = np.random.rand(len(queries), 384)  # Random 384-dim embeddings
        state["queries"] = queries
        state["query_embeddings"] = query_embeddings
        return state

    async def run(self, symbol: str) -> Dict[str, Any]:
        """Run the Market Intelligence workflow."""
        return await self.graph.ainvoke({"symbol": symbol, "market_data": {}, "summary": {}, "queries": {}})
    
    
agent = MarketAgent()
graph = agent.graph