from langchain.prompts import PromptTemplate
from market_intelligence.prompt.general_prompt import system_prompt, task_description, market_intelligence_effect_prompt



latest_outputformat = """
    Please ONLY return a valid JSON object. You MUST FOLLOW the JSON output format as follows:
    {
        "analysis": "ID: 000001 - Analysis that you provided for market intelligence 000001. ID: 000002 - Analysis that you provided for market intelligence 000002...",
        "summary": "The summary that you provided.",
        "query": {
            "short_term_query": "Query text that you provided for SHORT-TERM.",
            "medium_term_query": "Query text that you provided for MEDIUM-TERM.",
            "long_term_query": "Query text that you provided for LONG-TERM."
        }
    }
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


latest_prompt_template = f"""
    {system_prompt}

    {task_description}

    The following market intelligence (e.g., news, financial reports) contains latest (i.e., today)
information related to {{asset_symbol}}, including the corresponding dates, headlines, and contents, with each item
distinguished by a unique ID. Furthermore, if the day is not closed for trading, the section also provides the open, high,
low, close, and adjusted close prices.
    Latest market intelligence and prices are as follows: 
    {{latest_market_intelligence}}

    {market_intelligence_effect_prompt}


    {latest_prompt_summary}


    {latest_outputformat}

"""


def create_latest_prompt_template() -> PromptTemplate:





    return PromptTemplate(
        input_variables=["asset_name", "asset_symbol", "latest_market_intelligence"],
        template=latest_prompt_template,
    )


