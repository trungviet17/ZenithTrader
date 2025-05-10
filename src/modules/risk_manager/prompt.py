from langchain.prompts import PromptTemplate
from modules.risk_manager.state import TradeDecision, RiskProfile

def format_risk_profile(risk_profile):
    """Format risk profile into readable text"""
    formatted_text = f"""
The overall risk for this trade has been assessed as {risk_profile.overall_risk.value.upper()} with the following factor breakdown:
"""
    for factor, assessment in risk_profile.factor_assessments.items():
        factor_name = factor.value.replace('_', ' ').title()
        formatted_text += f"""
            - {factor_name} ({assessment.risk_level.value.upper()} risk, {assessment.confidence:.1%} confidence): 
            {assessment.reasoning}
            Suggested mitigations: {', '.join(assessment.mitigation_suggestions) if assessment.mitigation_suggestions else 'None'}
            """
    
    formatted_text += f"""
        Summary: {risk_profile.summary}

        Recommendations:
        {chr(10).join('- ' + rec for rec in risk_profile.recommendations)}
    """
    return formatted_text


def format_mitigation_plan(mitigation_plan):
    """Format mitigation plan into readable text"""
    sizing = mitigation_plan['position_sizing']
    stop_loss = mitigation_plan['stop_loss']
    
    formatted_text = f"""
        1. Position Sizing: Reduce quantity from {sizing['original_quantity']} to {sizing['suggested_quantity']} shares ({sizing['risk_adjustment']:.0%} of original risk exposure) due to {sizing['reasoning'].split(':')[1].strip()}.

        2. Stop Loss: Set stop loss at ${stop_loss['stop_loss_price']:.2f}, representing a {stop_loss['percentage']:.1%} buffer based on {stop_loss['reasoning'].split(':')[1].strip()}.

        3. Additional Recommendations:
        {chr(10).join('   - ' + rec for rec in mitigation_plan['additional_recommendations'])}

        Summary: {mitigation_plan['summary']}
        """
    return formatted_text



risk_reduction_prompt = """
       You are an advanced Risk Management AI specializing in financial trading analysis. Your task is to evaluate high-risk trade decisions, identify potential risks, and provide optimized alternatives.

        ### INPUT:
        You will receive details about a trading decision that has been flagged as HIGH or EXTREME risk, including:

        ```
        Trading Decision:
        - Asset Symbol: {trade_decision.symbol}
        - Action: {trade_decision.action}
        - Quantity: {trade_decision.quantity}
        - Price: ${trade_decision.price}
        - Agent: {trade_decision.agent_name}
        - Reasoning: {trade_decision.reasoning}
        - Exchange: {trade_decision.exchange_name}
        ```

        Risk Profile:
        {risk_profile}

        Mitigation Plan:
        {mitigation_plan}   
        ```

        ### YOUR TASK:
        1. Analyze the trade decision and associated risk profile
        2. Evaluate if the original decision is already well-aligned with the risk profile
        - If the original decision is already appropriate given the risk analysis, maintain it without changes
        - If significant risk issues exist, develop an optimized trading decision that reduces risk while preserving core elements of the investment thesis
        3. Provide clear reasoning for your assessment and any adjustments


        Be specific and quantitative in your recommendations. Include exact figures for adjusted quantities, prices, and risk parameters. Your goal is to transform high-risk trades into more prudent opportunities while preserving the core market thesis when possible.
        If the original trade is fundamentally unsound, you may recommend a complete reconsideration, but always attempt to find a risk-optimized version that captures part of the original investment idea.
        IMPORTANT: If your analysis determines that the original trading decision is already well-calibrated to the risk profile and no significant adjustments are needed, you should maintain the original decision parameters in your optimized_decision output and clearly state in the adjustment_reasoning that the original decision was appropriate. 
        In such cases, your confidence score should reflect your high level of agreement with the original decision.

        ### OUTPUT FORMAT:
        Return your analysis and recommendation in JSON format as follows:

        ```json
        {{
                "symbol": "Possibly modified asset symbol or original symbol if unchanged",
                "action": "Possibly modified buy/sell action or original action if unchanged",
                "quantity": "Possibly modified quantity or original quantity if unchanged",
                "price": "Possibly modified price or original price if unchanged",
                "reasoning": "New reasoning for optimized decision",
                "exchange_name": "Exchange name",
                "confidence": "Confidence score (0-1) indicating how much you agree with the original decision",
                "risk_profile": "Risk profile summary",
                "adjustment_reasoning": "Detailed explanation of why these changes were made, how they reduce risk, and how they maintain aspects of the original investment thesis (using just max 100 tokens for this part)",
        }}
        ```
"""



def get_risk_reduction_prompt(trade_decision, risk_profile, mitigation_plan):
    prompt = PromptTemplate(
        input_variables=["trade_decision", "risk_profile", "mitigation_plan"],
        template=risk_reduction_prompt
    )

    formatted_risk_profile = format_risk_profile(risk_profile)
    formatted_mitigation_plan = format_mitigation_plan(mitigation_plan)


    return prompt.format(
        trade_decision=trade_decision,
        risk_profile=formatted_risk_profile,
        mitigation_plan=formatted_mitigation_plan
    )







