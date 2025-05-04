import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agent.market_intelligence.prompt.general_prompt import system_prompt, task_description, market_intelligence_effect_prompt
from langchain.prompts import PromptTemplate
from server.schema import AssetData 

past_prompt_summary = """
     Based on the above information, you should analyze the key insights and summarize the market intelligence. Please
strictly follow the following constraints and output formats:

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

"""


past_outputformat = """
      You should return your analysis in the following format:
    
    {{
        "analysis": "- ID: 000001 - Analysis that you provided for market intelligence 000001. - ID: 000002 - Analysis that you provided for market intelligence 000002...",
        "summary": "The summary that you provided."
    }}

"""



def create_past_prompt_template(asset_data : AssetData) -> str:

    new_task_description = task_description.format(
        asset_type=asset_data.asset_type,
        asset_name=asset_data.asset_name,
        asset_symbol=asset_data.asset_symbol,
        asset_exchange=asset_data.asset_exchange,
        asset_sector=asset_data.asset_sector,
        asset_industry=asset_data.asset_industry,
        asset_description=asset_data.asset_description
    )

    new_past_prompt_summary = past_prompt_summary.format(
        asset_name=asset_data.asset_name,
        asset_symbol=asset_data.asset_symbol
    )


    past_prompt_template = f"""

        {system_prompt}

        {new_task_description}

        The following market intelligence (e.g., news, financial reports) contains past (i.e., before today)
    information related to {{asset_symbol}}, including the corresponding dates, headlines, and contents, with each item
    distinguished by a unique ID. Furthermore, if the day is not closed for trading, the section also provides the open, high,
    low, close, and adjusted close prices.

        

        Past market intelligence and prices are as follows: 
            {{past_market_intelligence}}

        {market_intelligence_effect_prompt}

        {new_past_prompt_summary}

        {past_outputformat}
    """
    


    return PromptTemplate(
        input_variables=[ "asset_symbol", "past_market_intelligence", "asset_name"],
        template=past_prompt_template,
    )

if __name__ == '__main__':
    def test_past_prompt_template():
        asset_data = AssetData(
            asset_type="stock",
            asset_name="Apple Inc.",
            asset_symbol="AAPL",
            asset_exchange="NASDAQ",
            asset_sector="Technology",
            asset_industry="Consumer Electronics",
            asset_description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
        )
        prompt = create_past_prompt_template(asset_data)

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



        formatted_prompt = prompt.format(
            past_market_intelligence=sample_intelligence,
            asset_symbol=asset_data.asset_symbol, 
            asset_name = asset_data.asset_name
        )

        print("Formatted Prompt:")
        print(formatted_prompt)


    test_past_prompt_template()