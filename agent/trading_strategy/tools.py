from typing import Dict, List, Any
import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
import requests

load_dotenv()


FMP_API_KEY = os.getenv("FMP_API_KEY")



def get_financial_metrics(ticker: str, end_date: str, limit=5) -> Dict[str, Any]:
  
    url = f"https://financialmodelingprep.com/stable/key-metrics?symbol={ticker}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            # Filter by end_date and sort by date descending
            filtered = [
                item for item in data
                if "date" in item and item["date"] <= end_date
            ]
            filtered.sort(key=lambda x: x["date"], reverse=True)
            return {"metrics": filtered[:limit]}
        else:
            return {"error": "Unexpected API response format", "response": data}
    except Exception as e:
        return {"error": str(e)}
    


def search_line_items(ticker: str, end_date: str, period="ttm", limit=5) -> Dict[str, Any]:

    url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&apikey={FMP_API_KEY}"

    try: 
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            # Filter by end_date and sort by date descending
            filtered = [
                item for item in data
                if "date" in item and item["date"] <= end_date
            ]
            filtered.sort(key=lambda x: x["date"], reverse=True)
            return {"financial_line_items": filtered[:limit]}
        else:
            return {"error": "Unexpected API response format", "response": data}
    
    except Exception as e:
        return {"error": str(e)}




def analyze_fundamentals(metrics: List) -> Dict[str, Any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    if latest_metrics.get("returnOnEquity") is not None:
        roe = latest_metrics["returnOnEquity"]
        if roe > 0.15:  # 15% ROE threshold
            score += 2
            reasoning.append(f"Strong ROE of {roe:.1%}")
        else:
            reasoning.append(f"Weak ROE of {roe:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    debt_to_equity = None
    
    if "totalDebt" in latest_metrics and "totalEquity" in latest_metrics and latest_metrics["totalEquity"] != 0:
        debt_to_equity = latest_metrics["totalDebt"] / latest_metrics["totalEquity"]
    
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            reasoning.append("Conservative debt levels")
        else:
            reasoning.append(f"High debt to equity ratio of {debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    operating_margin = None
    if latest_metrics.get("operatingMargin") is not None:
        operating_margin = latest_metrics["operatingMargin"]
    
    if operating_margin is not None:
        if operating_margin > 0.15:
            score += 2
            reasoning.append("Strong operating margins")
        else:
            reasoning.append(f"Weak operating margin of {operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    current_ratio = None
    if latest_metrics.get("currentRatio") is not None:
        current_ratio = latest_metrics["currentRatio"]
    elif "totalCurrentAssets" in latest_metrics and "totalCurrentLiabilities" in latest_metrics and latest_metrics["totalCurrentLiabilities"] != 0:
        current_ratio = latest_metrics["totalCurrentAssets"] / latest_metrics["totalCurrentLiabilities"]
    
    if current_ratio is not None:
        if current_ratio > 1.5:
            score += 1
            reasoning.append("Good liquidity position")
        else:
            reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {
        "score": score, 
        "details": "; ".join(reasoning), 
        "metrics": latest_metrics
    }




def analyze_consistency(financial_line_items: List) -> Dict[str, Any]:
    """Analyze earnings consistency and growth."""
    if not financial_line_items:
        return {"score": 0, "details": "No financial data provided"}
    
    if len(financial_line_items) < 4:  # Less than 4 periods available
        reasoning = ["Insufficient historical data for complete trend analysis"]
        score = 0
        
        # If at least one period is available, we can still provide some insights
        if len(financial_line_items) >= 1:
            latest = financial_line_items[0]
            
            # Check if net income is positive in the latest period
            if latest.get("netIncome") is not None and latest["netIncome"] > 0:
                score += 1
                reasoning.append(f"Positive net income of {latest['netIncome']:,.0f} in latest period")
            elif latest.get("netIncome") is not None:
                reasoning.append(f"Negative net income of {latest['netIncome']:,.0f} in latest period")
            else:
                reasoning.append("Net income data not available")
                
        return {"score": score, "details": "; ".join(reasoning)}
        
    # Original code continues here for 4+ periods
    score = 0
    reasoning = []

    # Check earnings growth trend
    earnings_values = [item.get("net_income") for item in financial_line_items if item.get("net_income")]
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate total growth rate from oldest to latest
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }



def analyze_moat(metrics: List) -> Dict[str, Any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    For simplicity, we look at stability of ROE/operating margins over multiple periods
    or high margin over the last few years. Higher stability => higher moat score.
    """
    if not metrics:
        return {"score": 0, "max_score": 3, "details": "Insufficient fundamental data"}

    reasoning = []
    moat_score = 0
    historical_roes = []
    historical_margins = []

    # Extract ROE and operating margin from available data
    for m in metrics:
        # Handle ROE
        if m.get("return_on_equity") is not None:
            historical_roes.append(m["return_on_equity"])
        elif m.get("returnOnEquity") is not None:
            historical_roes.append(m["returnOnEquity"])
        
        # Handle operating margin
        if m.get("operating_margin") is not None:
            historical_margins.append(m["operating_margin"])
        elif m.get("operatingMargin") is not None:
            historical_margins.append(m["operatingMargin"])
        # If operating margin isn't available, try to calculate it
        elif "enterpriseValue" in m and "evToSales" in m and "evToEBITDA" in m:
            try:
                revenue = m["enterpriseValue"] / m["evToSales"]
                ebitda = m["enterpriseValue"] / m["evToEBITDA"]
                # Operating margin is approximately EBITDA / Revenue
                calculated_margin = ebitda / revenue
                historical_margins.append(calculated_margin)
                reasoning.append(f"Operating margin calculated from EV metrics: {calculated_margin:.1%}")
            except (ZeroDivisionError, TypeError):
                reasoning.append("Could not calculate operating margin from available data")

    # If limited historical data is available, evaluate based on current values
    if len(historical_roes) < 3 and len(historical_roes) > 0:
        latest_roe = historical_roes[0]
        if latest_roe > 0.15:
            moat_score += 1
            reasoning.append(f"Strong current ROE of {latest_roe:.1%} (suggests potential moat)")
        else:
            reasoning.append(f"ROE of {latest_roe:.1%} below 15% threshold")
    elif len(historical_roes) >= 3:
        # Original code for 3+ periods of ROE data
        stable_roe = all(r > 0.15 for r in historical_roes)
        if stable_roe:
            moat_score += 1
            reasoning.append("Stable ROE above 15% across periods (suggests moat)")
        else:
            reasoning.append("ROE not consistently above 15%")
    else:
        reasoning.append("No ROE data available")

    # Similar adjustment for operating margin
    if len(historical_margins) < 3 and len(historical_margins) > 0:
        latest_margin = historical_margins[0]
        if latest_margin > 0.15:
            moat_score += 1
            reasoning.append(f"Strong current operating margin of {latest_margin:.1%} (moat indicator)")
        else:
            reasoning.append(f"Operating margin of {latest_margin:.1%} below 15% threshold")
    elif len(historical_margins) >= 3:
        # Original code for 3+ periods of margin data
        stable_margin = all(m > 0.15 for m in historical_margins)
        if stable_margin:
            moat_score += 1
            reasoning.append("Stable operating margins above 15% (moat indicator)")
        else:
            reasoning.append("Operating margin not consistently above 15%")
    else:
        reasoning.append("No operating margin data available")

    # If both are strong, add an extra point
    if moat_score == 2:
        moat_score += 1
        reasoning.append("Both ROE and margin strength indicate a solid moat")

    return {
        "score": moat_score,
        "max_score": 3,
        "details": "; ".join(reasoning),
    }


def analyze_management_quality(financial_line_items: List) -> Dict[str, Any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    
    # Check for share buybacks/dilution data
    if latest.get("issuance_or_purchase_of_equity_shares") is not None:
        if latest["issuance_or_purchase_of_equity_shares"] < 0:
            # Negative means the company spent money on buybacks
            mgmt_score += 1
            reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
        elif latest["issuance_or_purchase_of_equity_shares"] > 0:
            reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        # Alternative check using treasury stock or retained earnings if available
        if latest.get("treasuryStock") is not None and latest["treasuryStock"] > 0:
            mgmt_score += 1
            reasoning.append("Presence of treasury stock suggests share repurchases")
        elif latest.get("retainedEarnings") is not None:
            # Negative retained earnings might indicate aggressive buybacks
            if latest["retainedEarnings"] < 0 and latest.get("totalEquity", 0) > 0:
                mgmt_score += 1
                reasoning.append("Negative retained earnings with positive equity may indicate share repurchases")
            else:
                reasoning.append("No clear evidence of share repurchase program")
        else:
            reasoning.append("Insufficient data to evaluate share issuance/buybacks")

    # Check for dividend data
    if latest.get("dividends_and_other_cash_distributions") is not None:
        if latest["dividends_and_other_cash_distributions"] < 0:
            mgmt_score += 1
            reasoning.append("Company has a track record of paying dividends")
    else:
        # Try to infer dividend policy from other metrics
        if latest.get("earningsYield") is not None and latest.get("freeCashFlowYield") is not None:
            # Significant difference might indicate dividend payments
            if latest["earningsYield"] > 0 and latest["freeCashFlowYield"] > 0:
                mgmt_score += 0.5  # Partial score as this is inference
                reasoning.append("Positive earnings and free cash flow yield suggest potential for dividends")
        
        # Add information about limitations
        reasoning.append("Direct dividend data not available")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: List) -> Dict[str, Any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    if not financial_line_items or len(financial_line_items) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]

    net_income = latest.get("net_income")
    depreciation = latest.get("depreciation_and_amortization")
    capex = latest.get("capital_expenditure")

    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    # Estimate maintenance capex (typically 70-80% of total capex)
    maintenance_capex = capex * 0.75
    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"],
    }


def calculate_intrinsic_value(financial_line_items: List) -> Dict[str, Any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    if not financial_line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data for valuation"]}

    # Calculate owner earnings
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]

    # Get current market data
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.get("outstanding_shares")

    if not shares_outstanding:
        return {"intrinsic_value": None, "details": ["Missing shares outstanding data"]}

    # Buffett's DCF assumptions (conservative approach)
    growth_rate = 0.05  # Conservative 5% growth
    discount_rate = 0.09  # Typical ~9% discount rate
    terminal_multiple = 12
    projection_years = 10

    # Sum of discounted future owner earnings
    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    # Terminal value
    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)

    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }