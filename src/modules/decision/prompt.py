from langchain.prompts import PromptTemplate
from server.schema import AssetData

system_prompt = """
    You are an expert trader who have sufficient financial experience and provides expert guidance. Imagine working in a
    real market environment where you have access to various types of information (e.g., daily real-time market price, news,
    financial reports, professional investment guidance and market sentiment) relevant to financial markets. You will be able to view
    visual data that contains comprehensive information, including Kline charts accompanied by technical indicators, historical
    trading curves and cumulative return curves. And there will be some auxiliary strategies providing you with explanations for
    trading decisions. You are capable of deeply analyzing, understanding, and summarizing information, and use these information to
    make informed and wise trading decisions (i.e., BUY, HOLD and SELL).
    """

task_description = """
    You are currently targeting the trading of a company known as {asset_name}, which is denoted by the symbol
    {asset_symbol}. This corporation is publicly traded and is listed on the {asset_exchange}. Its primary operations are within
    the {asset_sector} sector, specifically within the {asset_industry} industry. To provide you with a better understanding,
    here is a brief description of {asset_name}: {asset_description}. In this role, your objective is to make correct trading
    decisions during the trading process of the asset represented by the {asset_symbol}, and considering step by step about the
    decision reasoning. To do so effectively, you will rely on a comprehensive set of information and data as follow
"""


output_rules = """
    Based on the above information, you should step-by-step analyze the summary of the market intelligence. And provide the
reasoning for what you should to BUY, SELL or HOLD on the asset. Please strictly follow the following constraints and output
formats:

    "reasoning": You should analyze step-by-step how the above information may affect the results of your decisions. You need to
follow the rules as follows and do not miss any of them:
    1. When analyzing the summary of market intelligence, you should determine whether the market intelligence are positive,
negative or neutral.
       - If the overall is neurtal, your decision should pay less attention to the summary of market intelligence.
       - If the overall is positive or negative. you should give a decision result based on this.
    2. When analyzing the analysis of price movements, you should determine whether the future trend is bullish or bearish and
reflect on the lessons you've learned.
       - If the future trend is bullish, you should consider a BUY instead of a HOLD to increase your profits.
       - If the future trend is bearish, you should consider a SELL instead of a HOLD to prevent further losses.
       - You should provide your decision result based on the analysis of price movements.
    3. When analyzing the analysis of the past trading decisions, you should reflect on the lessons you've learned.
       - If you have missed a BUY opportunity, you should BUY as soon as possible to increase your profits.
       - If you have missed a SELL, you should SELL immediately to prevent further losses.
       - You should provide your decision result based on the reflection of the past trading decisions.
    4. When analyzing the professional investment guidances, you should determine whether the guidances show the trend is bullish
or bearish. And provide your decision results.
    5. When analyzing the decisions and explanations of some trading strategies, you should consider the results and explanations
of their decisions together. And provide your decision results.
    6. When providing the final decision, you should pay less attention to the market intelligence whose sentiment is neutral or
unrelated.
    7. When providing the final decision, you should pay more attention to the market intelligence which will cause an immediate
impact on the price.
    8. When providing the final decision, if the overall market intelligence is mixed up, you should pay more attention to the
professional investment guidances, and consider which guidance is worthy trusting based on historical price.
    9. Before making a decision, you must check the current situation. If your CASH reserve is lower than the current Adj Close
Price, then the decision result should NOT be BUY. Similarly, the decision result should NOT be SELL if you have no existing
POSITION.
    10. Combining the results of all the above analysis and decisions, you should determine whether the current situation is
suitable for BUY, SELL or HOLD. And provide your final decision results.

    "reasoning": You should think step-by-step and provide the detailed reasoning to determine the decision result executed on
the current observation for the trading task. Please strictly follow the following constraints and output formats:
    1. You should provide the reasoning for each point of the "analysis" and the final results you provide.

    "action": Based on the above information and your analysis. Please strictly follow the following constraints and output
formats:
    1. You can only output one of BUY, HOLD and SELL.
    2. The above information may be in the opposite direction of decision-making (e.g., BUY or SELL), but you should consider step-
by-step all of the above information together to give an exact BUY or SELL decision result.
"""

output_format = """
    You should ONLY return a valid JSON object. You MUST FOLLOW the JSON output format as follows:
    {
        "reasoning": "Reason that you provided",
        "action": "BUY/HOLD/SELL",
        "quantity" : "The quantity of the asset you want to buy/sell must be a positive integer not less than 0",
        "price" : "The price of the asset you want to buy/sell must be a positive float not less than 0",
        "exchange_name": "The exchange name of the asset you want to buy/sell",
        "symbol": "The symbol of the asset you want to buy/sell",
        "confidence": "The confidence of the decision you made in range of 0 to 1",
    }
"""

decision_prompt = """
    {system_prompt}

    {task_description}


    The following are analysis of the latest (i.e., today) and past (i.e., before today) market intelligence
(e.g., news, financial reports) you've provided.

    The following is a analysis from your assistant of the past market intelligence and the latest market intelligence
        {market_intelligence}

    The analysis of price movements provided by your assistant across three time horizons: Short-Term, Medium-Term, and Long-Term.
    Past analysis and Latest analysis of price movements are as follows:
     +  {low_level_reflection}
    
    Please consider these reflections, identify the potential price movements patterns and characteristics of this
particular stock and incorporate these insights into your further analysis and reflections when applicable.

    As follows are the analysis provided by your assistant about the reflection on the trading decisions you
made during the trading processs, and evaluating if they were correct or incorrect, and considering if there are
opportunities for optimization to achieve maximum returns.
    Past and Latest reflections on the trading decisions are as follows:
        {high_level_reflection}

    Below are some decisions, market analyses from professional investors, and reasoning for their actions and analyses. You can use 
    these as reference to make your final investment decision:

    {trading_strategy}


    {output_rules}

    {output_format}

"""



def get_decision_prompt(market_intelligence: str,
                        low_level_reflection: str,
                        high_level_reflection: str,
                        trading_strategy: str,
                        asset_data: AssetData): 
    
    # Format the asset data into task description first
    task_desc = task_description.format(
        asset_name=asset_data.asset_name,
        asset_symbol=asset_data.asset_symbol,
        asset_exchange=asset_data.asset_exchange,
        asset_sector=asset_data.asset_sector,
        asset_industry=asset_data.asset_industry,
        asset_description=asset_data.asset_description,
    )

    # Create template with ALL variables as placeholders
    template = """
    {system_prompt}

    {task_desc}

    The following are analysis of the latest (i.e., today) and past (i.e., before today) market intelligence
    (e.g., news, financial reports) you've provided.

    The following is a analysis from your assistant of the past market intelligence and the latest market intelligence
        {market_intelligence}

    The analysis of price movements provided by your assistant across three time horizons: Short-Term, Medium-Term, and Long-Term.
    Past analysis and Latest analysis of price movements are as follows:
      {low_level_reflection}
    
    Please consider these reflections, identify the potential price movements patterns and characteristics of this
    particular stock and incorporate these insights into your further analysis and reflections when applicable.

    As follows are the analysis provided by your assistant about the reflection on the trading decisions you
    made during the trading processs, and evaluating if they were correct or incorrect, and considering if there are
    opportunities for optimization to achieve maximum returns.
    Past and Latest reflections on the trading decisions are as follows:
        {high_level_reflection}

    Below are some decisions, market analyses from professional investors, and reasoning for their actions and analyses. You can use 
    these as reference to make your final investment decision:

    {trading_strategy}

    {output_rules}

    {output_format}
    """

    # Create the prompt template with ALL variables
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "system_prompt",
            "task_desc",
            "market_intelligence",
            "low_level_reflection",
            "high_level_reflection",
            "trading_strategy",
            "output_rules",
            "output_format"
        ],
    )

    # Format with ALL variables at once
    return prompt.format(
        system_prompt=system_prompt,
        task_desc=task_desc,
        market_intelligence=market_intelligence,
        low_level_reflection=low_level_reflection,
        high_level_reflection=high_level_reflection,
        trading_strategy=trading_strategy,
        output_rules=output_rules,
        output_format=output_format
    )





