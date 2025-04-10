from market_intelligence.prompt.general_prompt import system_prompt, task_description, market_intelligence_effect_prompt



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


past_output_format = """
      You should return your analysis in the following format:
    
    {
        "analysis": "- ID: 000001 - Analysis that you provided for market intelligence 000001. - ID: 000002 - Analysis that you provided for market intelligence 000002...",
        "summary": "The summary that you provided."
    }

"""