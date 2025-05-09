from langchain.prompts import PromptTemplate

risk_analysis_prompt = """
        You are a Risk Management AI specializing in financial trading. 
        You need to analyze the following trade decision and its associated risk profile:

        Trade Decision:
        - Asset: {trade_decision.asset_symbol}
        - Action: {trade_decision.action}
        - Quantity: {trade_decision.quantity}
        - Price: ${trade_decision.price}
        - Agent: {trade_decision.agent_name}
        - Reasoning: {trade_decision.reasoning}

        Risk Profile:
        {risk_profile}

        Based on your analysis, determine if:
        1. The trade can be automatically approved (risks are acceptable)
        2. The trade requires human intervention (risks need review)
        3. The trade should be automatically rejected (risks are too high)

        Think step by step about market conditions, portfolio impact, and risk factors.
        Provide your reasoning and final decision.

        Decision (APPROVE, REVIEW, or REJECT): 
""" 

risk_reduction_prompt = """
        You are a sophisticated Risk Management AI specializing in financial trading. 
        The following trade decision has been identified as having HIGH or EXTREME risk:

        Trade Decision:
        - Asset: {trade_decision.asset_symbol}
        - Action: {trade_decision.action}
        - Quantity: {trade_decision.quantity}
        - Price: ${trade_decision.price}
        - Agent: {trade_decision.agent_name}
        - Reasoning: {trade_decision.reasoning}

        Risk Profile:
        {risk_profile}

        Mitigation Plan Generated:
        {mitigation_plan}

        I need you to think deeply about how this trade could be adjusted to significantly reduce risk while still capturing some of the original investment thesis.

        Please provide:

        1. SPECIFIC ADJUSTMENTS: Concrete modifications to the trade parameters (quantity, timing, entry price, etc.)
        2. ALTERNATIVE APPROACHES: Different ways to express the same market view with lower risk
        3. HEDGING STRATEGIES: Complementary trades that could offset specific risk factors
        4. CONDITIONAL EXECUTION PLAN: Specific market conditions or triggers that would make this trade more favorable

        For each suggestion, explain:
        - How it reduces the identified risks
        - What trade-offs it introduces
        - How it preserves parts of the original investment thesis

        Format your response in clear sections. Be specific and quantitative where possible.
"""








