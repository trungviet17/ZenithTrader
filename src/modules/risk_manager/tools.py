import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

from modules.risk_manager.state import TradeDecision, RiskProfile, RiskAssessment, RiskLevel, RiskFactor
import numpy as np
from typing import Dict, Any
from modules.risk_manager.api import get_volatility_data, get_liquidity_data, get_counterparty_data, get_concentration_data

def assess_market_volatility(trade_decision: TradeDecision, market_data: Dict[str, Any]) -> RiskAssessment:
    """Assess the risk related to market volatility for a given trade decision."""
   
    prices = np.array(market_data.get('prices', []))
    volumes = np.array(market_data.get('volumes', []))
    market_vix = market_data.get('market_vix')
    asset_beta = market_data.get('asset_beta', 1.0)
    implied_vol = market_data.get('implied_volatility')

    if len(prices) < 20:  
        return RiskAssessment(
            risk_level=RiskLevel.MEDIUM,  
            confidence=0.5,
            reasoning="Insufficient historical data to accurately assess volatility.",
            mitigation_suggestions=["Consider delaying trade until more market data is available"]
        )
    

    returns = np.diff(prices) / prices[:-1]
    hist_volatility_20d = np.std(returns[-20:]) * np.sqrt(252)  
    hist_volatility_10d = np.std(returns[-10:]) * np.sqrt(252)  

    avg_volume = np.mean(volumes[-20:])
    recent_volume = np.mean(volumes[-5:])
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

    price_range_ratio = (np.max(prices[-10:]) - np.min(prices[-10:])) / np.mean(prices[-10:])


    volatility_factors = []
    confidence_factors = []
    reasoning_points = []
    suggestions = []

    if hist_volatility_20d > 0.5:  # 50% annualized volatility
        volatility_factors.append(RiskLevel.EXTREME)
        confidence_factors.append(0.9)
        reasoning_points.append(f"Extremely high historical volatility ({hist_volatility_20d:.2%} annualized)")
        suggestions.append("Consider reducing position size significantly")
        suggestions.append("Use options to hedge downside risk")
    elif hist_volatility_20d > 0.3:
        volatility_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.85)
        reasoning_points.append(f"High historical volatility ({hist_volatility_20d:.2%} annualized)")
        suggestions.append("Reduce position size")
    elif hist_volatility_20d > 0.15:
        volatility_factors.append(RiskLevel.MEDIUM)
        confidence_factors.append(0.8)
        reasoning_points.append(f"Moderate historical volatility ({hist_volatility_20d:.2%} annualized)")
    else:
        volatility_factors.append(RiskLevel.LOW)
        confidence_factors.append(0.75)
        reasoning_points.append(f"Low historical volatility ({hist_volatility_20d:.2%} annualized)")


    if hist_volatility_10d > hist_volatility_20d * 1.5:
        volatility_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.8)
        reasoning_points.append("Recent volatility significantly higher than longer-term average")
        suggestions.append("Consider phasing the trade over multiple transactions")

    
    if volume_ratio > 2.0:
        volatility_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.7)
        reasoning_points.append(f"Unusual trading volume ({volume_ratio:.1f}x average)")
        suggestions.append("Monitor for news that could impact price") 

    if price_range_ratio > 0.2:  # 20% price range in last 10 periods
        volatility_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.75)
        reasoning_points.append(f"Large recent price swings (range of {price_range_ratio:.1%})")
        suggestions.append("Use limit orders instead of market orders")
    

    if market_vix is not None:
        if market_vix > 30:
            volatility_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Elevated market volatility index (VIX: {market_vix})")
        elif market_vix > 20:
            volatility_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Moderate market volatility index (VIX: {market_vix})")
    
    # Rule 6: Implied volatility if available
    if implied_vol is not None:
        if implied_vol > 0.5:
            volatility_factors.append(RiskLevel.EXTREME)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Options market anticipating extreme volatility ({implied_vol:.2%})")
        elif implied_vol > 0.3:
            volatility_factors.append(RiskLevel.HIGH) 
            confidence_factors.append(0.8)
            reasoning_points.append(f"Options market anticipating high volatility ({implied_vol:.2%})")
    
    # Rule 7: Beta-adjusted risk (if asset_beta is provided)
    if asset_beta > 1.5:
        volatility_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.75)
        reasoning_points.append(f"Asset has high beta ({asset_beta:.2f}), amplifying market movements")
        suggestions.append("Consider hedging with correlated assets")
    
    # Determine final risk level (take the highest risk level found)
    if RiskLevel.EXTREME in volatility_factors:
        final_risk_level = RiskLevel.EXTREME
    elif RiskLevel.HIGH in volatility_factors:
        final_risk_level = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in volatility_factors:
        final_risk_level = RiskLevel.MEDIUM
    else:
        final_risk_level = RiskLevel.LOW
    
    # Calculate confidence (average of all factors)
    average_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.7
    
    if final_risk_level == RiskLevel.EXTREME or final_risk_level == RiskLevel.HIGH:
        if "Consider using limit orders" not in suggestions:
            suggestions.append("Consider using limit orders")
        if "Phase the trade over multiple transactions" not in suggestions:
            suggestions.append("Phase the trade over multiple transactions")
    
    # Format the reasoning
    reasoning = ". ".join(reasoning_points) + "."
    
    return RiskAssessment(
        risk_level=final_risk_level,
        confidence=average_confidence,
        reasoning=reasoning,
        mitigation_suggestions=list(set(suggestions))  # Remove duplicates
    )
    
                           

def assess_liquidity_risk(trade_decision: TradeDecision, market_data: Dict[str, Any]) -> RiskAssessment:
    """Assess the risk related to liquidity for a given trade decision."""

    avg_daily_volume = market_data.get('avg_daily_volume', 0)
    bid_ask_spread = market_data.get('bid_ask_spread', 0.01)  # Default to 1%
    orderbook_depth = market_data.get('orderbook_depth', {})
    market_impact = market_data.get('market_impact_estimate')


    trade_quantity = trade_decision.quantity
    trade_value = trade_decision.price * trade_quantity

    liquidity_factors = []
    confidence_factors = []
    reasoning_points = [] 
    suggestions = []

    # rule 1 
    if avg_daily_volume > 0:
        volume_ratio = trade_quantity / avg_daily_volume
        
        if volume_ratio > 0.2:  # Trade is > 20% of daily volume
            liquidity_factors.append(RiskLevel.EXTREME)
            confidence_factors.append(0.9)
            reasoning_points.append(f"Trade represents {volume_ratio:.1%} of average daily volume")
            suggestions.append("Break order into smaller chunks over multiple days")
        elif volume_ratio > 0.1:  # Trade is > 10% of daily volume
            liquidity_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Trade represents {volume_ratio:.1%} of average daily volume")
            suggestions.append("Split order into multiple parts")
        elif volume_ratio > 0.05:  # Trade is > 5% of daily volume
            liquidity_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Trade represents {volume_ratio:.1%} of average daily volume")
            suggestions.append("Consider using TWAP/VWAP algorithms")
        else:
            liquidity_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Trade size is small relative to daily volume ({volume_ratio:.1%})")
    else:
        liquidity_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.6)
        reasoning_points.append("No volume data available to assess liquidity")
        suggestions.append("Research asset liquidity before proceeding")
    
    # rule 2 
    if bid_ask_spread > 0.05:  # > 5% spread
        liquidity_factors.append(RiskLevel.EXTREME)
        confidence_factors.append(0.9)
        reasoning_points.append(f"Very wide bid-ask spread of {bid_ask_spread:.1%}")
        suggestions.append("Use limit orders only")
        suggestions.append("Consider alternative assets with better liquidity")
    elif bid_ask_spread > 0.02:  # > 2% spread
        liquidity_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.85)
        reasoning_points.append(f"Wide bid-ask spread of {bid_ask_spread:.1%}")
        suggestions.append("Use limit orders")
    elif bid_ask_spread > 0.01:  # > 1% spread
        liquidity_factors.append(RiskLevel.MEDIUM)
        confidence_factors.append(0.8)
        reasoning_points.append(f"Moderate bid-ask spread of {bid_ask_spread:.1%}")
    else:
        liquidity_factors.append(RiskLevel.LOW)
        confidence_factors.append(0.85)
        reasoning_points.append(f"Tight bid-ask spread of {bid_ask_spread:.1%}")


    if market_impact is not None:
        if market_impact > 0.03:  # > 3% price impact
            liquidity_factors.append(RiskLevel.EXTREME)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Estimated market impact of {market_impact:.1%}")
            suggestions.append("Use algorithmic execution to minimize impact")
        elif market_impact > 0.01:  # > 1% price impact
            liquidity_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Estimated market impact of {market_impact:.1%}")


    if RiskLevel.EXTREME in liquidity_factors:
        final_risk_level = RiskLevel.EXTREME
    elif RiskLevel.HIGH in liquidity_factors:
        final_risk_level = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in liquidity_factors:
        final_risk_level = RiskLevel.MEDIUM
    else:
        final_risk_level = RiskLevel.LOW
    
    # Calculate confidence (average of all factors)
    average_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.7
    
    # Format the reasoning
    reasoning = ". ".join(reasoning_points) + "."
    
    return RiskAssessment(
        risk_level=final_risk_level,
        confidence=average_confidence,
        reasoning=reasoning,
        mitigation_suggestions=list(set(suggestions))  # Remove duplicates
    )


def assess_counterparty_risk(trade_decision: TradeDecision, exchange_data: Dict[str, Any]) -> RiskAssessment:
    """Assess counterparty risk for a given trade decision."""

    exchange_name = exchange_data.get('symbol', 'Unknown')
    is_regulated = exchange_data.get('is_regulated', False)
    jurisdiction = exchange_data.get('jurisdiction', 'Unknown')
    insurance_coverage = exchange_data.get('insurance_coverage', 0)
    credit_rating = exchange_data.get('credit_rating')
    security_incidents = exchange_data.get('security_incidents', [])
    

    risk_factors = []
    confidence_factors = []
    reasoning_points = []
    suggestions = []


    if is_regulated: 
        risk_factors.append(RiskLevel.LOW)
        confidence_factors.append(0.9)
        reasoning_points.append(f"Trading on regulated exchange ({exchange_name}) under {jurisdiction} jurisdiction")
    else : 
        risk_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.85)
        reasoning_points.append(f"Trading on unregulated exchange ({exchange_name})")
        suggestions.append("Consider using a regulated exchange/broker")
        suggestions.append("Limit exposure to this counterparty")
    

    # rule 2 
    trade_value = trade_decision.quantity * trade_decision.price
    if insurance_coverage > 0:
        coverage_ratio = insurance_coverage / trade_value
        if coverage_ratio < 1:
            risk_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.7)
            reasoning_points.append(f"Insurance coverage ({insurance_coverage:,.2f}) less than trade value")
            suggestions.append("Consider splitting funds across multiple accounts/exchanges")
    else:
        risk_factors.append(RiskLevel.HIGH)
        confidence_factors.append(0.75)
        reasoning_points.append("No insurance coverage information available")

    # rule 3 
    if credit_rating:
        if credit_rating.startswith('A'):  # AAA, AA, A
            risk_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Strong counterparty credit rating ({credit_rating})")
        elif credit_rating.startswith('B'):  # BBB, BB, B
            if credit_rating.startswith('BBB'):
                risk_factors.append(RiskLevel.LOW)
                confidence_factors.append(0.75)
                reasoning_points.append(f"Adequate counterparty credit rating ({credit_rating})")
            else:
                risk_factors.append(RiskLevel.MEDIUM)
                confidence_factors.append(0.8)
                reasoning_points.append(f"Moderate counterparty credit rating ({credit_rating})")
                suggestions.append("Monitor counterparty financial health")
        else:  # CCC or below
            risk_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Low counterparty credit rating ({credit_rating})")
            suggestions.append("Minimize exposure to this counterparty")

    # rule 4 
    if security_incidents:
        recent_incidents = [i for i in security_incidents if i.get('years_ago', 99) < 3]
        if recent_incidents:
            risk_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Counterparty has had {len(recent_incidents)} security incidents in the past 3 years")
            suggestions.append("Use enhanced security measures (2FA, withdrawal limits)")
            suggestions.append("Consider alternative exchanges with better security history")
        elif security_incidents:
            risk_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.75)
            reasoning_points.append(f"Counterparty has had security incidents, but none recently")
            suggestions.append("Use standard security best practices")
    else:
        risk_factors.append(RiskLevel.LOW)
        confidence_factors.append(0.7)
        reasoning_points.append("No known security incidents for this counterparty")

    if RiskLevel.EXTREME in risk_factors:
        final_risk_level = RiskLevel.EXTREME
    elif RiskLevel.HIGH in risk_factors:
        final_risk_level = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in risk_factors:
        final_risk_level = RiskLevel.MEDIUM
    else:
        final_risk_level = RiskLevel.LOW
    
    # Calculate confidence (average of all factors)
    average_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.7
    
    # Format the reasoning
    reasoning = ". ".join(reasoning_points) + "."
    
    return RiskAssessment(
        risk_level=final_risk_level,
        confidence=average_confidence,
        reasoning=reasoning,
        mitigation_suggestions=list(set(suggestions))  # Remove duplicates
    )




def assess_concentration_risk(trade_decision: TradeDecision, portfolio_data: Dict[str, Any]) -> RiskAssessment:
    
    total_portfolio_value = portfolio_data.get('total_value', 0)
    holdings = portfolio_data.get('holdings', {})
    sector_allocation = portfolio_data.get('sector_allocation', {})
    correlations = portfolio_data.get('correlations', {})


    asset_symbol = trade_decision.symbol
    trade_value = trade_decision.quantity * trade_decision.price
    asset_sector = portfolio_data.get('asset_sector', {}).get(asset_symbol, 'Unknown')


    risk_factors = []
    confidence_factors = []
    reasoning_points = []
    suggestions = []


    # rule 1 
    if total_portfolio_value > 0:
    
        current_position_value = 0
        if asset_symbol in holdings:
            current_position_value = holdings[asset_symbol].get('market_value', 0)
        
        if trade_decision.action.lower() == 'buy':
            new_position_value = current_position_value + trade_value
        elif trade_decision.action.lower() == 'sell':
            new_position_value = max(0, current_position_value - trade_value)
        else:
            new_position_value = current_position_value
        
        position_percentage = new_position_value / total_portfolio_value if total_portfolio_value > 0 else 0
        
        if position_percentage > 0.25: 
            risk_factors.append(RiskLevel.EXTREME)
            confidence_factors.append(0.9)
            reasoning_points.append(f"Position will represent {position_percentage:.1%} of portfolio (extreme concentration)")
            suggestions.append("Consider reducing position size")
            suggestions.append("Implement strict stop-loss to manage outsized position")
        elif position_percentage > 0.15:  
            risk_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Position will represent {position_percentage:.1%} of portfolio (high concentration)")
            suggestions.append("Consider capping position at 15% of portfolio")
        elif position_percentage > 0.05: 
            risk_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Position will represent {position_percentage:.1%} of portfolio (moderate concentration)")
        else:
            risk_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Position will represent only {position_percentage:.1%} of portfolio (well diversified)")


    # rule 2 
    if asset_sector != 'Unknown' and sector_allocation:
        current_sector_allocation = sector_allocation.get(asset_sector, 0)
        sector_allocation_after_trade = current_sector_allocation
        
        if total_portfolio_value > 0:
            if trade_decision.action.lower() == 'buy':
                sector_allocation_after_trade = current_sector_allocation + (trade_value / total_portfolio_value)
            elif trade_decision.action.lower() == 'sell':
                sector_allocation_after_trade = current_sector_allocation - (trade_value / total_portfolio_value)
        
        if sector_allocation_after_trade > 0.4: 
            risk_factors.append(RiskLevel.EXTREME)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Sector {asset_sector} exposure will increase to {sector_allocation_after_trade:.1%} (extreme concentration)")
            suggestions.append("Consider diversifying into different sectors")
        elif sector_allocation_after_trade > 0.3:  # Sector > 30% of portfolio
            risk_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Sector {asset_sector} exposure will increase to {sector_allocation_after_trade:.1%} (high concentration)")
            suggestions.append("Consider capping sector exposure")
        elif sector_allocation_after_trade > 0.2:  # Sector > 20% of portfolio
            risk_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.75)
            reasoning_points.append(f"Sector {asset_sector} exposure will increase to {sector_allocation_after_trade:.1%} (moderate concentration)")
            suggestions.append("Monitor sector exposure")
        else:
            risk_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Sector {asset_sector} exposure will be {sector_allocation_after_trade:.1%} (well diversified)")
    
    # rule 3 
    high_correlation_assets = 0
    avg_correlation = 0
    
    if correlations:
        correlation_values = list(correlations.values())
        if correlation_values:
            avg_correlation = sum(correlation_values) / len(correlation_values)
            high_correlation_assets = sum(1 for c in correlation_values if c > 0.7)
        
        if avg_correlation > 0.7:
            risk_factors.append(RiskLevel.HIGH)
            confidence_factors.append(0.8)
            reasoning_points.append(f"Asset highly correlated (avg {avg_correlation:.2f}) with existing portfolio")
            suggestions.append("Consider assets with lower correlation to portfolio")
        elif avg_correlation > 0.5:
            risk_factors.append(RiskLevel.MEDIUM)
            confidence_factors.append(0.75)
            reasoning_points.append(f"Asset moderately correlated (avg {avg_correlation:.2f}) with existing portfolio")
        elif avg_correlation > 0:
            risk_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.7)
            reasoning_points.append(f"Asset has low correlation (avg {avg_correlation:.2f}) with existing portfolio")
        elif avg_correlation < 0:
            risk_factors.append(RiskLevel.LOW)
            confidence_factors.append(0.85)
            reasoning_points.append(f"Asset negatively correlated (avg {avg_correlation:.2f}) with existing portfolio (beneficial for diversification)")
    

    if RiskLevel.EXTREME in risk_factors:
        final_risk_level = RiskLevel.EXTREME
    elif RiskLevel.HIGH in risk_factors:
        final_risk_level = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in risk_factors:
        final_risk_level = RiskLevel.MEDIUM
    else:
        final_risk_level = RiskLevel.LOW

    average_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.7
    

    reasoning = ". ".join(reasoning_points) + "."
    
    return RiskAssessment(
        risk_level=final_risk_level,
        confidence=average_confidence,
        reasoning=reasoning,
        mitigation_suggestions=list(set(suggestions))  # Remove duplicates
    )


def compile_risk_profile(trade_decision: TradeDecision, vol_market_data, lig_market_data ,  portfolio_data: Dict[str, Any], exchange_data: Dict[str, Any]) -> RiskProfile:
   
    factor_assessments = {
        RiskFactor.MARKET_VOLATILITY: assess_market_volatility(trade_decision, vol_market_data),
        RiskFactor.LIQUIDITY: assess_liquidity_risk(trade_decision, lig_market_data),
        RiskFactor.COUNTERPARTY: assess_counterparty_risk(trade_decision, exchange_data),
        RiskFactor.CONCENTRATION: assess_concentration_risk(trade_decision, portfolio_data),
    }
    
    risk_levels = [assessment.risk_level for assessment in factor_assessments.values()]
    if RiskLevel.EXTREME in risk_levels:
        overall_risk = RiskLevel.EXTREME
    elif RiskLevel.HIGH in risk_levels:
        overall_risk = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in risk_levels:
        overall_risk = RiskLevel.MEDIUM
    else:
        overall_risk = RiskLevel.LOW
    
    # Compile recommendations
    all_recommendations = []
    for assessment in factor_assessments.values():
        all_recommendations.extend(assessment.mitigation_suggestions)
    
    # Create summary
    summary = f"Overall risk assessment for {trade_decision.action.upper()} {trade_decision.quantity} {trade_decision.symbol} " \
              f"at ${trade_decision.price}: {overall_risk.name} risk level."
    
    return RiskProfile(
        overall_risk=overall_risk,
        factor_assessments=factor_assessments,
        summary=summary,
        recommendations=list(set(all_recommendations))  # Deduplicate recommendations
    )


if __name__ == "__main__":

    from pprint import pprint

    ticker = "AAPL" 

    holdings = {
        "AAPL": 10,
        "MSFT": 5,
        "GOOGL": 8
    }

    vol_data = get_volatility_data(ticker)
    liq_data = get_liquidity_data(ticker)

    counterparty_data = get_counterparty_data("NASDAQ")
    concentration_data = get_concentration_data(holdings)


    print("Volatility Data: ", vol_data)
    print("Liquidity Data: ", liq_data)
    print("Counterparty Data: ", counterparty_data)
    print("Concentration Data: ", concentration_data)

    
    pprint(compile_risk_profile(
        TradeDecision(
            symbol="AAPL",
            action="buy",
            quantity=10,
            price=150.0
        ),
        vol_data,
        liq_data,
        concentration_data,
        counterparty_data
    ))