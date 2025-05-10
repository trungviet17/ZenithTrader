from typing import List, Dict, Any


def analyze_value_metrics(metrics: List) -> Dict[str, Any]:
    """Graham focused on absolute valuation metrics as primary indicators."""
    if not metrics:
        return {"score": 0, "details": "Insufficient value metrics data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.get("price_to_earnings_ratio"):
        pe_ratio = latest_metrics["price_to_earnings_ratio"]
        if pe_ratio < 15:
            score += 2
            reasoning.append(f"Low P/E ratio of {pe_ratio:.1f} (below Graham's threshold of 15)")
        else:
            reasoning.append(f"High P/E ratio of {pe_ratio:.1f} (above Graham's threshold of 15)")
    else:
        reasoning.append("P/E ratio data not available")

    if latest_metrics.get("price_to_book_ratio"):
        pb_ratio = latest_metrics["price_to_book_ratio"]
        if pb_ratio < 1.5:
            score += 2
            reasoning.append(f"Low P/B ratio of {pb_ratio:.1f} (below Graham's threshold of 1.5)")
        else:
            reasoning.append(f"High P/B ratio of {pb_ratio:.1f} (above Graham's threshold of 1.5)")
    else:
        reasoning.append("P/B ratio data not available")

    if latest_metrics.get("payout_ratio"):
        payout_ratio = latest_metrics["payout_ratio"]
        if payout_ratio > 0.2:  # Assuming a decent dividend payout
            score += 1
            reasoning.append(f"Good dividend payout ratio of {payout_ratio:.1%}")
        else:
            reasoning.append(f"Low dividend payout ratio of {payout_ratio:.1%}")
    else:
        reasoning.append("Dividend payout data not available")

    return {
        "score": score,
        "details": "; ".join(reasoning),
        "metrics": latest_metrics
    }


def analyze_safety_margin(metrics: List, financial_line_items: List) -> Dict[str, Any]:
    """Graham's margin of safety analysis."""
    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for safety margin analysis"}

    latest_metrics = metrics[0]
    latest_financials = financial_line_items[0]
    score = 0
    reasoning = []

    try:
        if latest_metrics.get("earnings_per_share") and latest_metrics.get("book_value_per_share"):
            eps = latest_metrics["earnings_per_share"]
            bvps = latest_metrics["book_value_per_share"]
            
            if eps > 0 and bvps > 0:
                graham_number = (22.5 * eps * bvps) ** 0.5
                
                # Current price approximation from market cap and shares outstanding
                if latest_metrics.get("market_cap") and latest_financials.get("outstanding_shares"):
                    price = latest_metrics["market_cap"] / latest_financials["outstanding_shares"]
                    
                    # Margin of safety calculation
                    margin = (graham_number - price) / price
                    
                    if margin > 0.5:  # More than 50% below Graham's number
                        score += 3
                        reasoning.append(f"Excellent margin of safety: {margin:.1%} below Graham's number")
                    elif margin > 0.2:  # More than 20% below Graham's number
                        score += 2
                        reasoning.append(f"Good margin of safety: {margin:.1%} below Graham's number")
                    elif margin > 0:  # At least below Graham's number
                        score += 1
                        reasoning.append(f"Minimal margin of safety: {margin:.1%} below Graham's number")
                    else:
                        reasoning.append(f"No margin of safety: {-margin:.1%} above Graham's number")
                else:
                    reasoning.append("Unable to calculate current price")
            else:
                reasoning.append("Negative earnings or book value")
        else:
            reasoning.append("Missing earnings per share or book value data")
    except Exception as e:
        reasoning.append(f"Error calculating Graham's number: {str(e)}")

    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def analyze_financial_strength(metrics: List) -> Dict[str, Any]:
    """Graham emphasized financial strength and stability."""
    if not metrics:
        return {"score": 0, "details": "Insufficient financial strength data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.get("current_ratio"):
        current_ratio = latest_metrics["current_ratio"]
        if current_ratio > 2:
            score += 2
            reasoning.append(f"Strong current ratio of {current_ratio:.1f} (above Graham's threshold of 2)")
        elif current_ratio > 1.5:
            score += 1
            reasoning.append(f"Adequate current ratio of {current_ratio:.1f}")
        else:
            reasoning.append(f"Weak current ratio of {current_ratio:.1f} (below Graham's standard)")
    else:
        reasoning.append("Current ratio data not available")

    if latest_metrics.get("debt_to_assets"):
        debt_ratio = latest_metrics["debt_to_assets"]
        if debt_ratio < 0.3:
            score += 2
            reasoning.append(f"Low debt-to-assets ratio of {debt_ratio:.1f}")
        elif debt_ratio < 0.5:
            score += 1
            reasoning.append(f"Moderate debt-to-assets ratio of {debt_ratio:.1f}")
        else:
            reasoning.append(f"High debt-to-assets ratio of {debt_ratio:.1f}")
    else:
        reasoning.append("Debt-to-assets data not available")

    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def analyze_earnings_stability(metrics: List, years: int = 5) -> Dict[str, Any]:
    """Graham required earnings stability over time."""
    if not metrics or len(metrics) < years:
        return {"score": 0, "details": f"Insufficient data for {years}-year earnings stability analysis"}

    score = 0
    reasoning = []
    
    earnings_history = []
    for i, period in enumerate(metrics):
        if i >= years:
            break
        if period.get("earnings_per_share") is not None:
            earnings_history.append(period["earnings_per_share"])

    if earnings_history and all(eps > 0 for eps in earnings_history):
        score += 2
        reasoning.append(f"Positive earnings in all {len(earnings_history)} analyzed periods")
    elif earnings_history and any(eps > 0 for eps in earnings_history):
        reasoning.append(f"Some positive earnings, but not consistent in all {len(earnings_history)} periods")
    else:
        reasoning.append("No positive earnings in analyzed periods")

    if len(earnings_history) >= 2:
        growth_periods = sum(1 for i in range(len(earnings_history)-1) if earnings_history[i] > earnings_history[i+1])
        growth_percentage = growth_periods / (len(earnings_history)-1)
        
        if growth_percentage >= 0.7:
            score += 2
            reasoning.append(f"Strong earnings growth in {growth_percentage:.0%} of periods")
        elif growth_percentage >= 0.5:
            score += 1
            reasoning.append(f"Moderate earnings growth in {growth_percentage:.0%} of periods")
        else:
            reasoning.append(f"Weak earnings growth in only {growth_percentage:.0%} of periods")

    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def calculate_graham_intrinsic_value(metrics: List, financial_line_items: List) -> Dict[str, Any]:
    """Calculate Graham's intrinsic value using his formula."""
    if not metrics or not financial_line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data for Graham valuation"]}

    latest_metrics = metrics[0]
    latest_financials = financial_line_items[0]
    
    try:
        # Graham's intrinsic value formula: V = EPS × (8.5 + 2g)
        # where V is intrinsic value, EPS is earnings per share, and g is growth rate
        
        if not latest_metrics.get("earnings_per_share"):
            return {"intrinsic_value": None, "details": ["Earnings per share data not available"]}
            
        eps = latest_metrics["earnings_per_share"]
        
        # Estimate growth rate from historical data if available
        growth_rate = 0.0
        if latest_metrics.get("earnings_growth"):
            growth_rate = max(0, min(latest_metrics["earnings_growth"], 0.15))  # Cap between 0-15%
        
        # Calculate intrinsic value
        intrinsic_value = eps * (8.5 + 2 * growth_rate * 100)
        
        # Total shares for total company value
        if latest_financials.get("outstanding_shares"):
            total_value = intrinsic_value * latest_financials["outstanding_shares"]
        else:
            total_value = None
            
        return {
            "intrinsic_value": total_value,
            "per_share_value": intrinsic_value,
            "eps_used": eps,
            "growth_rate_used": growth_rate,
            "details": ["Graham formula: V = EPS × (8.5 + 2g)"]
        }
    except Exception as e:
        return {"intrinsic_value": None, "details": [f"Error in calculation: {str(e)}"]}