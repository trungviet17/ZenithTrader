import numpy as np


class AugmentedTools:
    """Tools for augmenting the decision-making process."""
    
    def __init__(self):
        pass
    
    def get_macd_signal(self, price_data):
        """Get MACD crossover trading signal."""
        # Simplified MACD calculation
        close_prices = price_data['close'].values
        ema12 = self._calculate_ema(close_prices, 12)
        ema26 = self._calculate_ema(close_prices, 26)
        macd_line = ema12 - ema26
        signal_line = self._calculate_ema(macd_line, 9)
        
        # Generate trading signal
        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            return "BUY"  # MACD line crosses above signal line
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            return "SELL"  # MACD line crosses below signal line
        else:
            return "HOLD"  # No crossover
    
    def get_kdj_rsi_signal(self, price_data):
        """Get KDJ with RSI filter trading signal."""
        # Simplified KDJ calculation
        high_prices = price_data['high'].values
        low_prices = price_data['low'].values
        close_prices = price_data['close'].values
        
        # Calculate RSI
        rsi = self._calculate_rsi(close_prices, 14)
        
        # Calculate KDJ
        k, d, j = self._calculate_kdj(high_prices, low_prices, close_prices)
        
        # Generate trading signal with RSI filter
        if k[-1] > d[-1] and k[-2] <= d[-2] and rsi[-1] < 70:
            return "BUY"  # K line crosses above D line and not overbought
        elif k[-1] < d[-1] and k[-2] >= d[-2] and rsi[-1] > 30:
            return "SELL"  # K line crosses below D line and not oversold
        else:
            return "HOLD"  # No valid signal
    
    def get_z_score_mean_reversion_signal(self, price_data):
        """Get Z-score mean reversion trading signal."""
        # Calculate Z-score
        close_prices = price_data['close'].values
        mean = np.mean(close_prices[-20:])  # 20-day mean
        std = np.std(close_prices[-20:])
        z_score = (close_prices[-1] - mean) / std
        
        # Generate trading signal
        if z_score < -2:
            return "BUY"  # Significantly below mean, expect reversion upward
        elif z_score > 2:
            return "SELL"  # Significantly above mean, expect reversion downward
        else:
            return "HOLD"  # Within normal range
    
    def _calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def _calculate_kdj(self, high, low, close, n=9, m1=3, m2=3):
        """Calculate KDJ indicator."""
        rsv = np.zeros_like(close)
        k = np.zeros_like(close)
        d = np.zeros_like(close)
        j = np.zeros_like(close)
        
        for i in range(n-1, len(close)):
            period_high = np.max(high[i-n+1:i+1])
            period_low = np.min(low[i-n+1:i+1])
            
            if period_high != period_low:
                rsv[i] = (close[i] - period_low) / (period_high - period_low) * 100
            else:
                rsv[i] = 50
            
            if i == n-1:
                k[i] = d[i] = rsv[i]
            else:
                k[i] = (m1 * k[i-1] + rsv[i]) / (m1 + 1)
                d[i] = (m2 * d[i-1] + k[i]) / (m2 + 1)
            
            j[i] = 3 * k[i] - 2 * d[i]
        
        return k, d, j