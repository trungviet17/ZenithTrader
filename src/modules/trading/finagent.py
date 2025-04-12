from decision_making.decision import AugmentedTools, DecisionMakingModule
from reflection.highlevel_reflection import HighLevelReflectionModule, LowLevelReflectionModule
from decision_making.llm import MockLLMService
from market_intelligence.market_intelligence import MarketIntelligenceModule
from modules.memory.manager import MemoryModule


class FinAgent:
    """Main FinAgent class integrating all modules."""
    
    def __init__(self, llm_service=None):
        # Initialize LLM service
        self.llm_service = llm_service if llm_service else MockLLMService()
        
        # Initialize modules
        self.memory_module = MemoryModule()
        self.market_intelligence_module = MarketIntelligenceModule(self.memory_module, self.llm_service)
        self.low_level_reflection_module = LowLevelReflectionModule(self.memory_module, self.llm_service)
        self.high_level_reflection_module = HighLevelReflectionModule(self.memory_module, self.llm_service)
        self.augmented_tools = AugmentedTools()
        self.decision_making_module = DecisionMakingModule(self.llm_service, self.augmented_tools)
        
        # Trading history
        self.trading_history = []
    
    def process_data(self, date, news, price_data, kline_chart, trading_chart, trader_preference="Moderate"):
        """Process data for a single trading day and make a decision."""
        # 1. Process market intelligence
        market_intelligence = self.market_intelligence_module.process_latest_market_intelligence(
            date, news, price_data
        )
        
        # 2. Calculate price movements
        price_movements = self._calculate_price_movements(price_data)
        
        # 3. Low-level reflection
        low_level_reflection = self.low_level_reflection_module.reflect(
            date, market_intelligence, kline_chart, price_movements
        )
        
        # 4. High-level reflection
        past_actions = self.trading_history[-14:] if len(self.trading_history) > 0 else []
        high_level_reflection = self.high_level_reflection_module.reflect(
            date, market_intelligence, low_level_reflection, trading_chart, past_actions
        )
        
        # 5. Make decision
        decision = self.decision_making_module.make_decision(
            date, market_intelligence, low_level_reflection, high_level_reflection, 
            price_data, trader_preference
        )
        
        # 6. Update trading history
        self.trading_history.append({
            'date': date,
            'action': decision['action'],
            'reasoning': decision['reasoning']
        })
        
        return decision
    
    def _calculate_price_movements(self, price_data):
        """Calculate price movements for different time frames."""
        # For demonstration purposes, using simple calculations
        # In a real implementation, these would be more sophisticated
        
        close_prices = price_data['close'].values
        
        # Short-term: 1-day change
        short_term_change = 0
        if len(close_prices) > 1:
            short_term_change = (close_prices[-1] / close_prices[-2] - 1) * 100
        
        # Medium-term: 5-day change
        medium_term_change = 0
        if len(close_prices) > 5:
            medium_term_change = (close_prices[-1] / close_prices[-6] - 1) * 100
        
        # Long-term: 20-day change
        long_term_change = 0
        if len(close_prices) > 20:
            long_term_change = (close_prices[-1] / close_prices[-21] - 1) * 100
        
        return {
            'short_term': f"Over the past 1 day, the price movement ratio has shown a change of {short_term_change:.2f}%.",
            'medium_term': f"Over the past 5 days, the price movement ratio has shown a change of {medium_term_change:.2f}%.",
            'long_term': f"Over the past 20 days, the price movement ratio has shown a change of {long_term_change:.2f}%."
        }