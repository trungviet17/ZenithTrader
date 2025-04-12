class MockLLMService:
    """Mock LLM service for generating responses."""
    
    def __init__(self, model_type="gpt-4"):
        self.model_type = model_type
    
    def generate(self, prompt):
        """Generate a response to the given prompt."""
        # In a real implementation, this would call an LLM API
        # For this mock, we'll return predefined responses based on the prompt content
        
        if "Latest market intelligence" in prompt:
            return {
                'analysis': "Analysis of market intelligence shows mixed signals with some positive developments in product announcements but concerns about regulatory challenges.",
                'summary': "Overall market sentiment appears cautiously optimistic with an expectation of moderate growth in the medium term.",
                'query': "Product announcements, regulatory challenges, market growth expectations"
            }
        elif "price movements" in prompt:
            return {
                'reasoning': "The price movements can be attributed to recent product announcements which created initial excitement, followed by profit-taking and some concerns about regulatory challenges.",
                'query': "Price movements following product announcements, regulatory impact on stock price"
            }
        elif "Trading decisions" in prompt:
            return {
                'reasoning': "The BUY decision on the previous trading day appears to have been correct as the price increased by 2.5%, validating the analysis of positive market sentiment.",
                'improvement': "Future decisions could be improved by setting clear stop-loss levels to protect against unexpected downturns.",
                'summary': "Successful decisions were based on comprehensive analysis of market intelligence and technical indicators, with attention to both short-term catalysts and medium-term trends.",
                'query': "Successful trading decisions, stop-loss strategies, market sentiment analysis"
            }
        elif "step-by-step analyze" in prompt:
            # Determine action based on signals in the prompt
            if "MACD Signal: BUY" in prompt and "positive" in prompt:
                action = "BUY"
            elif "MACD Signal: SELL" in prompt and "negative" in prompt:
                action = "SELL"
            else:
                action = "HOLD"
                
            return {
                'analysis': "Market intelligence suggests a cautiously optimistic outlook, with technical indicators showing mixed signals. Recent price movements indicate potential for upward momentum in the short term.",
                'reasoning': "The combination of positive market sentiment, favorable technical indicators (particularly MACD and RSI), and lessons from past successful trades suggests that a BUY position is warranted at this time.",
                'action': action
            }
        else:
            return {
                'summary': "Generated response for prompt: " + prompt[:50] + "..."
            }