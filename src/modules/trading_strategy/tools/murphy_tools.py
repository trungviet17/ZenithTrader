import pandas as pd 
import numpy as np 
from typing import Dict, Any



def calculate_ma(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    

    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    return df 



def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    # cal rsi 
    delta = df['close'].diff(1) 
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # macd 
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']


    window = 14 
    df['lowest_low'] = df['low'].rolling(window=window).min()
    df['highest_high'] = df['high'].rolling(window=window).max()
    df['%K'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
    df['%D'] = df['%K'].rolling(window=3).mean()

    return df 


def calculate_sup_res(df: pd.DataFrame, window : int = 10) -> pd.DataFrame:
    if df.empty or len(df) < window * 2:
        return pd.DataFrame()
    
    pivots = []
    sup = []
    res = []

    for i in range(window, len(df) - window):

        if all(df['high'].iloc[i] > df['high'].iloc[i - j] for j in range(1, window + 1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i + j] for j in range(1, window + 1)):
            
            pivots.append((df.index[i], df['high'].iloc[i], 'resistance'))
            res.append(df['high'].iloc[i])

        elif all(df['low'].iloc[i] < df['low'].iloc[i - j] for j in range(1, window + 1)) and \
            all(df['low'].iloc[i] < df['low'].iloc[i + j] for j in range(1, window + 1)):
                
            pivots.append((df.index[i], df['low'].iloc[i], 'support'))
            sup.append(df['low'].iloc[i])


    def cluster_levels(levels: list, threshold: float = 0.01) -> list:
        """
        nhom cac level lai voi nhau neu chung co cung mot threshold
        """
        if not levels:
            return []

        result = []
        levels = sorted(levels)
        
        current_cluster = [levels[0]]
        for value in levels[1:]:
            if value <= current_cluster[-1] * (1 + threshold):
                current_cluster.append(value)
            else:
                result.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [value]
        
        if current_cluster:
            result.append(sum(current_cluster) / len(current_cluster))
        
        return result
    
    current_price = df['close'].iloc[-1]

    support_level = cluster_levels([s for s in sup if s < current_price])
    resistance_level = cluster_levels([r for r in res if r > current_price])

    return {
        "support": support_level[-3:] if len(support_level) > 3 else support_level,
        "resistance": resistance_level[:3], 
        "current_price": current_price,
    }


def analyze_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phan tich volume 
    
    """

    if df.empty:
        return pd.DataFrame()


    recent_days = 5 
    if len(df) < recent_days + 20: 
        return {'volume_trend' : 'unk', 'price_volumn_divergence': False, "details" : "No data available"}
    
    
    recent_volume = df['volume'].iloc[-recent_days:].mean()
    historical_volume = df['volume'].iloc[-30:-recent_days].mean()



    volume_change = (recent_volume - historical_volume) / historical_volume
    if volume_change > 0.2:
        volume_trend = "strongly increasing"
    elif volume_change > 0.05:
        volume_trend = "increasing"
    elif volume_change < -0.2:
        volume_trend = "strongly decreasing"
    elif volume_change < -0.05:
        volume_trend = "decreasing"
    else:
        volume_trend = "stable"

    recent_price = df['close'].iloc[-1]
    recent_price_change = (recent_price - df['close'].iloc[~recent_days]) / df['close'].iloc[~recent_days]
    price_trend = "up" if recent_price_change > 0 else "down"
    price_volume_divergence = (price_trend == "up" and volume_trend in ["decreasing", "strongly decreasing"]) or \
                              (price_trend == "down" and volume_trend in ["increasing", "strongly increasing"])
                              
    details = []
    if volume_trend in ["strongly increasing", "increasing"] and price_trend == "up":
        details.append("Strong volume confirms price uptrend")
    elif volume_trend in ["strongly decreasing", "decreasing"] and price_trend == "down":
        details.append("Declining volume confirms price downtrend")
    elif price_volume_divergence:
        if price_trend == "up":
            details.append("Bearish divergence: price rising on declining volume")
        else:
            details.append("Bullish divergence: price falling on increasing volume")
    
    
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    recent_obv_change = (df['obv'].iloc[-1] - df['obv'].iloc[-recent_days]) / abs(df['obv'].iloc[-recent_days])
    
    if abs(recent_obv_change) > 0.05:
        direction = "up" if recent_obv_change > 0 else "down"
        details.append(f"On-balance volume trending {direction}")
    
    return {
        "volume_trend": volume_trend,
        "price_volume_divergence": price_volume_divergence,
        "details": "; ".join(details)
    }


def analyze_trend(df: pd.DataFrame) : 

    if df.empty or 'SMA_20' not in df.columns or 'SMA_50' not in df.columns or 'SMA_200' not in df.columns:
        return {"primary_trend": "unknown", "details": "Insufficient data for trend analysis"}
    
    latest = df.iloc[-1]
    
    trend_score = 0
    reasons = []
    
    close = latest['close']
    if close > latest['SMA_20']:
        trend_score += 1
        reasons.append("Price above 20-day MA")
    else:
        trend_score -= 1
        reasons.append("Price below 20-day MA")
        
    if close > latest['SMA_50']:
        trend_score += 2
        reasons.append("Price above 50-day MA")
    else:
        trend_score -= 2
        reasons.append("Price below 50-day MA")
        
    if close > latest['SMA_200']:
        trend_score += 3
        reasons.append("Price above 200-day MA")
    else:
        trend_score -= 3
        reasons.append("Price below 200-day MA")
    
    # Moving average alignments
    if latest['SMA_20'] > latest['SMA_50']:
        trend_score += 1
        reasons.append("20-day MA above 50-day MA")
    else:
        trend_score -= 1
        reasons.append("20-day MA below 50-day MA")
        
    if latest['SMA_50'] > latest['SMA_200']:
        trend_score += 2
        reasons.append("50-day MA above 200-day MA")
    else:
        trend_score -= 2
        reasons.append("50-day MA below 200-day MA")
    
    # Determine trend strength and direction
    if trend_score >= 7:
        primary_trend = "strong uptrend"
    elif trend_score >= 3:
        primary_trend = "uptrend"
    elif trend_score > 0:
        primary_trend = "weak uptrend"
    elif trend_score == 0:
        primary_trend = "neutral/sideways"
    elif trend_score >= -3:
        primary_trend = "weak downtrend"
    elif trend_score >= -7:
        primary_trend = "downtrend"
    else:
        primary_trend = "strong downtrend"
    
    # Calculate trend slope
    if len(df) >= 20:
        slope_20 = (df['SMA_20'].iloc[-1] - df['SMA_20'].iloc[-10]) / df['SMA_20'].iloc[-10]
        slope_info = f"20-day MA slope: {slope_20:.2%} over past 10 days"
        reasons.append(slope_info)
    
    return {
        "primary_trend": primary_trend,
        "trend_score": trend_score,
        "details": "; ".join(reasons)
    }


def identify_patterns(df: pd.DataFrame): 
    """
    Phan tich patterns cua chart 
    """

    if df.empty or len(df) < 30:
        return {"patterns": [], "details": "Insufficient data for pattern analysis"}
    
    patterns = []
    details = []
    recent = df.tail(30)
    
    lows = recent['low'].values
    recent_mins = []
    
    for i in range(1, len(lows)-1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            recent_mins.append((i, lows[i]))
    
    if len(recent_mins) >= 2:
    
        for i in range(len(recent_mins)-1):
            for j in range(i+1, len(recent_mins)):
                idx1, low1 = recent_mins[i]
                idx2, low2 = recent_mins[j]
                
                if abs(idx1 - idx2) >= 5 and abs(low1 - low2) / low1 < 0.03:
                    patterns.append("double bottom")
                    details.append("Potential double bottom pattern (bullish reversal)")
                    break
    
    if len(recent_mins) >= 3:
        for i in range(len(recent_mins)-2):
            idx1, low1 = recent_mins[i]
            idx2, low2 = recent_mins[i+1]
            idx3, low3 = recent_mins[i+2]
            
            # Head should be lower than shoulders
            if low2 < low1 and low2 < low3 and abs(low1 - low3) / low1 < 0.05:
                patterns.append("head and shoulders")
                details.append("Potential head and shoulders pattern (bearish reversal)")
    
    # Check for price channels
    highs = recent['high'].values
    lows = recent['low'].values
    
    high_slope = np.polyfit(np.arange(len(highs)), highs, 1)[0]
    low_slope = np.polyfit(np.arange(len(lows)), lows, 1)[0]
    
    if abs(high_slope - low_slope) < 0.0005:  # Parallel slopes
        if high_slope > 0.001:
            patterns.append("ascending channel")
            details.append("Price moving in ascending channel (bullish continuation)")
        elif high_slope < -0.001:
            patterns.append("descending channel")
            details.append("Price moving in descending channel (bearish continuation)")
        else:
            patterns.append("horizontal channel")
            details.append("Price moving in horizontal channel (consolidation)")
    
    return {
        "patterns": patterns,
        "details": "; ".join(details) if details else "No clear patterns detected"
    }







def analyze_momentum(df: pd.DataFrame) -> Dict[str, Any]:
    """phan tich momentum """
    if df.empty or 'RSI' not in df.columns or 'MACD_line' not in df.columns:
        return {"momentum_signal": "unknown", "details": "Insufficient data for momentum analysis"}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    
    momentum_score = 0
    reasons = []
    
    # RSI analysis
    if latest['RSI'] < 30:
        momentum_score += 2
        reasons.append(f"RSI oversold at {latest['RSI']:.1f}")
    elif latest['RSI'] < 40:
        momentum_score += 1
        reasons.append(f"RSI approaching oversold at {latest['RSI']:.1f}")
    elif latest['RSI'] > 70:
        momentum_score -= 2
        reasons.append(f"RSI overbought at {latest['RSI']:.1f}")
    elif latest['RSI'] > 60:
        momentum_score -= 1
        reasons.append(f"RSI approaching overbought at {latest['RSI']:.1f}")
    
    # MACD analysis
    if prev is not None:
        if latest['MACD_line'] > latest['MACD_signal'] and prev['MACD_line'] <= prev['MACD_signal']:
            momentum_score += 2
            reasons.append("MACD bullish crossover (signal to buy)")
        elif latest['MACD_line'] < latest['MACD_signal'] and prev['MACD_line'] >= prev['MACD_signal']:
            momentum_score -= 2
            reasons.append("MACD bearish crossover (signal to sell)")
            
    if latest['MACD_line'] > 0:
        momentum_score += 1
        reasons.append("MACD line above zero (bullish)")
    else:
        momentum_score -= 1
        reasons.append("MACD line below zero (bearish)")
    
    # Stochastic analysis
    if '%K' in df.columns and '%D' in df.columns:
        if latest['%K'] < 20:
            momentum_score += 1
            reasons.append(f"Stochastic oversold at {latest['%K']:.1f}")
        elif latest['%K'] > 80:
            momentum_score -= 1
            reasons.append(f"Stochastic overbought at {latest['%K']:.1f}")
            
        if prev is not None:
            if latest['%K'] > latest['%D'] and prev['%K'] <= prev['%D']:
                momentum_score += 1
                reasons.append("Stochastic bullish crossover")
            elif latest['%K'] < latest['%D'] and prev['%K'] >= prev['%D']:
                momentum_score -= 1
                reasons.append("Stochastic bearish crossover")
    
    # Determine overall momentum signal
    if momentum_score >= 3:
        momentum_signal = "strongly bullish"
    elif momentum_score > 0:
        momentum_signal = "bullish"
    elif momentum_score == 0:
        momentum_signal = "neutral"
    elif momentum_score > -3:
        momentum_signal = "bearish"
    else:
        momentum_signal = "strongly bearish"
    
    return {
        "momentum_signal": momentum_signal,
        "momentum_score": momentum_score,
        "details": "; ".join(reasons)
    }




