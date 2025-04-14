import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


from langchain.prompts import PromptTemplate
from agent.market_intelligence.prompt.general_prompt import system_prompt, task_description, market_intelligence_effect_prompt
from server.schema import AssetData


latest_outputformat = """
    Please ONLY return a valid JSON object. You MUST FOLLOW the JSON output format as follows:  
    {{
        "analysis": "ID: 000001 - Analysis that you provided for market intelligence 000001. ID: 000002 - Analysis that you provided for market intelligence 000002...",
        "summary": "The summary that you provided.",
        "query": {{
            "short_term_query": "Query text that you provided for SHORT-TERM.",
            "medium_term_query": "Query text that you provided for MEDIUM-TERM.",
            "long_term_query": "Query text that you provided for LONG-TERM."
        }}
    }}
"""


latest_prompt_summary = """
    Based on the above information, you should analyze the key insights and summarize the market intelligence.
    Please strictly follow the following constraints and output formats:

    "analysis": This field is used to extract key insights from the above information. You should analyze step-by-step and
    follow the rules as follows and do not miss any of them:
    1. Please disregard UNRELATED market intelligence.
    2. For each piece of market intelligence, you should analyze it and extract key insights according to the following steps:
       - Extract the key insights that can represent this market intelligence. It should NOT contain IDs, {asset_name} or
         {asset_symbol}.
       - Analyze the market effects duration and provide the duration of the effects on asset prices. You are only allowed to select
         the only one of the three types: SHORT-TERM, MEDIUM-TERM and LONG-TERM.
       - Analyze the market sentiment and provide the type of market sentiment. A clear preference over POSITIVE or NEGATIVE is much
         better than being NEUTRAL. You are only allowed to select the only one of the three types: POSITIVE, NEGATIVE and NEUTRAL.
    3. The analysis you provide for each piece of market intelligence should be concise and clear, with no more than 40 tokens per
       piece.
    4. Your analysis MUST be in the following format:
       - ID: 000001 - Analysis that you provided for market intelligence 000001.
       - ID: 000002 - Analysis that you provided for market intelligence 000002.
       - ...

    "summary": This field is used to summarize the above analysis and extract key investment insights. You should summarize
    step-by-step and follow the rules as follows and do not miss any of them:
    1. Please disregard UNRELATED market intelligence.
    2. Because this field is primarily used for decision-making in trading tasks, you should focus primarily on asset related key
       investment insights.
    3. Please combine and summarize market intelligence on similar sentiment tendencies and duration of effects on asset prices.
    4. You should provide an overall analysis of all the market intelligence, explicitly provide a market sentiment (POSITIVE,
       NEGATIVE or NEUTRAL) and provide a reasoning for the analysis.
    5. Summary that you provided for market intelligence should contain IDs (e.g., ID: 000001, 000002).
    6. The summary you provide should be concise and clear, with no more than 300 tokens.

    "query": This field will be used to retrieve past market intelligence based on the duration of effects on asset prices. You
    should summarize step-by-step the above analysis and extract key insights. Please follow the rules as follows and do not miss
    any of them:
    1. Please disregard UNRELATED market intelligence.
    2. Because this field is primarily used for retrieving past market intelligence based on the duration of effects on asset
       prices, you should focus primarily on asset related key insights and duration of effects.
    3. Please combine the analysis of market intelligence on similar duration of effects on asset prices.
    4. You should provide a query text for each duration of effects on asset prices, which can be associated with several pieces of
       market intelligence.
       - The query text that you provide should be primarily keywords from the original market intelligence contained.
       - The query text that you provide should NOT contain IDs, {asset_name} or {asset_symbol}.
       - The query text that you provide should be concise and clear, with no more than 100 tokens per query
"""





def create_latest_prompt_template(asset_data : AssetData) -> PromptTemplate:
    
    new_task_description = task_description.format(
        asset_type=asset_data.asset_type,
        asset_name=asset_data.asset_name,
        asset_symbol=asset_data.asset_symbol,
        asset_exchange=asset_data.asset_exchange,
        asset_sector=asset_data.asset_sector,
        asset_industry=asset_data.asset_industry,
        asset_description=asset_data.asset_description
    )

    

    new_latest_prompt_summary = latest_prompt_summary.format(
        asset_name=asset_data.asset_name,
        asset_symbol=asset_data.asset_symbol
    )



    latest_prompt_template = f"""
        {system_prompt}

        {new_task_description}

        The following market intelligence (e.g., news, financial reports) contains latest (i.e., today)
    information related to {{asset_symbol}}, including the corresponding dates, headlines, and contents, with each item
    distinguished by a unique ID. Furthermore, if the day is not closed for trading, the section also provides the open, high,
    low, close, and adjusted close prices.
        Latest market intelligence and prices are as follows: 
        {{latest_market_intelligence}}


        You can use the following tools to retrieve latest market intelligence and prices for {{asset_symbol}} 
        if all latest market intelligence is not enough to analyze the market intelligence:

        1. Use the "web_search" tool to find general market information about {{asset_symbol}}
        2. Use the "news_search" tool to retrieve recent news articles about {{asset_symbol}}
        3. Use the "financial_report" tool to obtain comprehensientiment for {{asset_symbol}}
        5. Use the "get_price" tool to get historical price datave financial data for {{asset_symbol}}
        4. Use the "sentiment_analysis" tool to analyze market s for {{asset_symbol}}

        {market_intelligence_effect_prompt}


        {new_latest_prompt_summary}


        {latest_outputformat}

    """


    return PromptTemplate(
        input_variables=["asset_name", "asset_symbol", "latest_market_intelligence"],
        template=latest_prompt_template,
    )


if __name__ == '__main__': 


    def test_latest_prompt_template():
        # Create sample asset data
        sample_asset = AssetData(
            asset_type="stock",
            asset_name="Apple Inc.",
            asset_symbol="AAPL",
            asset_exchange="NASDAQ",
            asset_sector="Technology",
            asset_industry="Consumer Electronics",
            asset_description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
        )
        
        # Create the prompt template
        prompt_template = create_latest_prompt_template(sample_asset)
        
        # Sample market intelligence data for testing
        sample_intelligence = """
        ID: 000001
        Date: 2025-04-12
        Headline: Apple announces new iPhone model
        Content: Apple Inc. revealed their latest iPhone model with groundbreaking AI features today, expected to boost sales in Q3.
        
        ID: 000002
        Date: 2025-04-12
        Headline: Supply chain issues resolved
        Content: Apple confirmed that recent supply chain constraints have been resolved, allowing for improved production capacity.
        
        Price Data:
        Open: 198.45
        High: 203.78
        Low: 197.90
        Close: 202.56
        Volume: 73,452,890
        """
        
        # Format the prompt with sample data
        formatted_prompt = prompt_template.format(
            asset_name=sample_asset.asset_name,
            asset_symbol=sample_asset.asset_symbol, 
            latest_market_intelligence=sample_intelligence
        )
        
        # Print the formatted prompt to verify
        print("=== FORMATTED PROMPT ===")
        print(formatted_prompt)
        print("=== END OF FORMATTED PROMPT ===")
        
        # Simple assertion to ensure the prompt contains key elements
        assert sample_asset.asset_name in formatted_prompt
        assert sample_asset.asset_symbol in formatted_prompt
        assert "ID: 000001" in formatted_prompt
        assert "groundbreaking AI features" in formatted_prompt
        
        print("All tests passed successfully!")

    test_latest_prompt_template()


