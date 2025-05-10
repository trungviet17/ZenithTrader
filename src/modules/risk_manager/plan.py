import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.risk_manager.state import TradeDecision, RiskLevel, RiskProfile
from typing import Dict, Any



def suggest_position_sizing(trade_decision: TradeDecision, risk_profile: RiskProfile) -> Dict[str, Any]:
   
    risk_factor = {
        RiskLevel.LOW: 1.0,
        RiskLevel.MEDIUM: 0.75,
        RiskLevel.HIGH: 0.5,
        RiskLevel.EXTREME: 0.25
    }

    risk_adjustment = risk_factor[risk_profile.overall_risk]
    suggested_quantity = int(trade_decision.quantity * risk_adjustment)


    return {
        "original_quantity": trade_decision.quantity,
        "suggested_quantity": suggested_quantity,
        "risk_adjustment": risk_adjustment,
        "reasoning": f"Adjusted quantity based on overall risk level: {risk_profile.overall_risk}"
    }


def suggest_stop_loss(trade_decision: TradeDecision, risk_profile: RiskProfile) -> Dict[str, Any]:
    stop_loss_factor = {
        RiskLevel.LOW: 0.05,
        RiskLevel.MEDIUM: 0.07,
        RiskLevel.HIGH: 0.1,
        RiskLevel.EXTREME: 0.15
    }

    percentage = stop_loss_factor[risk_profile.overall_risk]

    if trade_decision.action == "buy": 
        stop_loss_price = trade_decision.price * (1 - percentage)
    else :
        stop_loss_price = trade_decision.price * (1 + percentage)


    return {
        "stop_loss_price": stop_loss_price,
        "percentage": percentage,
        "reasoning": f"Stop loss set at {percentage * 100}% based on overall risk level: {risk_profile.overall_risk}"
    }



def generate_mitigation_plan(trade_decision: TradeDecision, risk_profile: RiskProfile) -> Dict[str, Any]:


    position_sizing = suggest_position_sizing(trade_decision, risk_profile)
    stop_loss = suggest_stop_loss(trade_decision, risk_profile)

    mitigation_plan = {
        "position_sizing": position_sizing,
        "stop_loss": stop_loss,
        "additional_recommendations": risk_profile.recommendations,
        "summary": f"Risk mitigation plan for {trade_decision.action} {trade_decision.symbol}: "
                   f"Adjust position to {position_sizing['suggested_quantity']} shares, "
                   f"set stop loss at ${stop_loss['stop_loss_price']}."

    }

    return mitigation_plan