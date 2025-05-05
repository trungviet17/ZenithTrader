from agent.market_intelligence.graph import create_graph, create_sample_vectorstore
from agent.trading_strategy.agents.buffett import create_buffett_agent



# vectorstore = create_sample_vectorstore(AssetData(
#     asset_name="Apple Inc.",
#     asset_symbol="AAPL",
#     asset_type="Stock",
#     asset_exchange="NASDAQ",
#     asset_sector="Technology",
#     asset_industry="Consumer Electronics",
#     asset_description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
# ))


graph = create_buffett_agent()
