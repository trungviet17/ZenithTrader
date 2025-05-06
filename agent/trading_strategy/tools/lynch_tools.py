from typing import Dict, List, Any
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def analyze_growth(metrics: List) -> Dict[str, Any]:
    """Lynch focused heavily on earnings growth."""
    if not metrics or len(metrics) < 3:
        return {"score": 0, "details": "Insufficient growth data", "category": None}

    score = 0
    reasoning = []
    earnings_growth_rates = []
    revenue_growth_rates = []
    
    for period in metrics:
        if period.get("earnings_growth") is not None:
            earnings_growth_rates.append(period["earnings_growth"])
        if period.get("revenue_growth") is not None:
            revenue_growth_rates.append(period["revenue_growth"])
 
    avg_earnings_growth = sum(earnings_growth_rates) / len(earnings_growth_rates) if earnings_growth_rates else None
    avg_revenue_growth = sum(revenue_growth_rates) / len(revenue_growth_rates) if revenue_growth_rates else None
    
    company_category = None
    
    if avg_earnings_growth is not None:
        if avg_earnings_growth > 0.25:  # 25%+ growth
            company_category = "Fast Grower"
            score += 3
            reasoning.append(f"Fast Grower with exceptional earnings growth of {avg_earnings_growth:.1%}")
        elif 0.10 <= avg_earnings_growth <= 0.25:  # 10-25% growth
            company_category = "Stalwart"
            score += 2
            reasoning.append(f"Stalwart with solid earnings growth of {avg_earnings_growth:.1%}")
        elif 0 < avg_earnings_growth < 0.10:  # 0-10% growth
            company_category = "Slow Grower"
            score += 1
            reasoning.append(f"Slow Grower with modest earnings growth of {avg_earnings_growth:.1%}")
        else:  # Negative growth
            company_category = "Turnaround/Cyclical"
            reasoning.append(f"Potentially a Turnaround or Cyclical stock with declining earnings ({avg_earnings_growth:.1%})")
    
    if len(earnings_growth_rates) >= 3:
        positive_growth_periods = sum(1 for rate in earnings_growth_rates if rate > 0)
        consistency_pct = positive_growth_periods / len(earnings_growth_rates)
        
        if consistency_pct > 0.8:
            score += 2
            reasoning.append(f"Consistent earnings growth in {consistency_pct:.1%} of periods")
        elif consistency_pct > 0.5:
            score += 1
            reasoning.append(f"Moderate earnings consistency with growth in {consistency_pct:.1%} of periods")
        else:
            reasoning.append(f"Inconsistent earnings with growth in only {consistency_pct:.1%} of periods")
    
    if avg_revenue_growth is not None:
        if avg_revenue_growth > 0.15:
            score += 2
            reasoning.append(f"Strong revenue growth of {avg_revenue_growth:.1%}")
        elif avg_revenue_growth > 0.05:
            score += 1
            reasoning.append(f"Moderate revenue growth of {avg_revenue_growth:.1%}")
        else:
            reasoning.append(f"Weak revenue growth of {avg_revenue_growth:.1%}")

    return {
        "score": score,
        "details": "; ".join(reasoning),
        "category": company_category,
        "avg_earnings_growth": avg_earnings_growth,
        "avg_revenue_growth": avg_revenue_growth
    }


def analyze_peg_ratio(metrics: List) -> Dict[str, Any]:
    """Lynch famously used the PEG ratio as a key valuation metric."""
    if not metrics:
        return {"score": 0, "details": "Insufficient PEG data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.get("peg_ratio"):
        peg = latest_metrics["peg_ratio"]
        
        if peg < 0:
            reasoning.append(f"Negative PEG ratio ({peg:.2f}) indicates earnings decline")
        elif peg < 0.5:
            score += 3
            reasoning.append(f"Excellent PEG ratio of {peg:.2f} (well below Lynch's threshold of 1)")
        elif peg < 1.0:
            score += 2
            reasoning.append(f"Good PEG ratio of {peg:.2f} (below Lynch's threshold of 1)")
        elif peg < 1.5:
            score += 1
            reasoning.append(f"Fair PEG ratio of {peg:.2f}")
        else:
            reasoning.append(f"Expensive PEG ratio of {peg:.2f} (above Lynch's threshold of 1)")
    else:
        reasoning.append("PEG ratio data not available")

    if latest_metrics.get("price_to_earnings_ratio") and latest_metrics.get("earnings_growth"):
        pe = latest_metrics["price_to_earnings_ratio"]
        growth = latest_metrics["earnings_growth"] * 100  # Convert to percentage
        
        if pe < growth:
            score += 2
            reasoning.append(f"P/E ({pe:.1f}) is below growth rate ({growth:.1f}%) - attractive by Lynch's standard")
        elif pe < growth * 1.5:
            score += 1
            reasoning.append(f"P/E ({pe:.1f}) is reasonably valued relative to growth ({growth:.1f}%)")
        else:
            reasoning.append(f"P/E ({pe:.1f}) exceeds growth rate ({growth:.1f}%) significantly")

    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def analyze_competitive_advantage(metrics: List) -> Dict[str, Any]:
    """Lynch favored companies with sustainable competitive advantages."""
    if not metrics:
        return {"score": 0, "details": "Insufficient data for competitive analysis"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.get("operating_margin"):
        margin = latest_metrics["operating_margin"]
        
        if margin > 0.2:
            score += 2
            reasoning.append(f"High operating margin of {margin:.1%} suggests strong competitive position")
        elif margin > 0.1:
            score += 1
            reasoning.append(f"Decent operating margin of {margin:.1%}")
        else:
            reasoning.append(f"Low operating margin of {margin:.1%}")

    if latest_metrics.get("return_on_equity"):
        roe = latest_metrics["return_on_equity"]
        
        if roe > 0.20:
            score += 2
            reasoning.append(f"Excellent ROE of {roe:.1%} indicates strong competitive advantage")
        elif roe > 0.15:
            score += 1
            reasoning.append(f"Good ROE of {roe:.1%}")
        else:
            reasoning.append(f"Average or below average ROE of {roe:.1%}")

    if latest_metrics.get("asset_turnover"):
        turnover = latest_metrics["asset_turnover"]
        
        if turnover > 1.5:
            score += 1
            reasoning.append(f"Efficient asset utilization with turnover of {turnover:.1f}")
        else:
            reasoning.append(f"Asset turnover of {turnover:.1f}")

    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def get_industry_data(ticker: str) -> Dict[str, Any]:
    """Lynch emphasized understanding the industry context."""

    try:
        industry_data = {
            "industry_growth": 0.05,  # 5% industry growth example
            "market_size": 500,  # $500B market size example
            "competitive_landscape": "moderate",
            "barriers_to_entry": "medium",
            "industry_trends": [
                "Increasing digital transformation",
                "Growing focus on sustainability",
                "Rising regulatory scrutiny"
            ]
        }
        
        return industry_data
        
    except Exception as e:
        return {"error": str(e)}