class MarketIntelligenceModule:
    """Module for collecting, analyzing, and summarizing market intelligence."""
    
    def __init__(self, memory_module, llm_service):
        self.memory_module = memory_module
        self.llm_service = llm_service
    
    def process_latest_market_intelligence(self, date, news, price_data):
        """Process latest market intelligence and store in memory."""
        prompt = f"""Latest market intelligence and prices are as follows:
        Date: {date}
        News: {news}
        Price Data: {price_data}
        
        Based on the above information, you should analyze the key insights and summarize the market intelligence. Please strictly follow the following constraints and output formats:
        "analysis": This field is used to extract key insights from the above information.
        "summary": This field is used to summarize the analysis and extract key investment insights.
        "query": This field will be used to retrieve past market intelligence. Please include separate queries for short-term, medium-term, and long-term retrieval.
        """
        
        response = self.llm_service.generate(prompt)
        analysis, summary, query_text = self._parse_response(response)
        
        # Store in memory
        self.memory_module.store_market_intelligence(date, news, summary, query_text)
        
        # Retrieve past market intelligence
        past_mi = self.memory_module.retrieve_market_intelligence(query_text)
        
        # Process past market intelligence
        past_summary = self._process_past_market_intelligence(past_mi)
        
        return {
            'analysis': analysis,
            'summary': summary,
            'query_text': query_text,
            'past_summary': past_summary
        }
    
    def _parse_response(self, response):
        """Parse LLM response to extract analysis, summary, and query text."""
        # Simple parsing logic - in real implementation this would be more robust
        analysis = response.get('analysis', '')
        summary = response.get('summary', '')
        query_text = response.get('query', '')
        
        return analysis, summary, query_text
    
    def _process_past_market_intelligence(self, past_mi):
        """Process past market intelligence to extract insights."""
        if not past_mi:
            return ""
        
        # Combine past market intelligence summaries
        combined = " ".join([item['summary'] for item in past_mi])
        
        prompt = f"""Past market intelligence summaries are as follows:
        {combined}
        
        Please summarize these past market intelligence insights in a concise way that would be helpful for making trading decisions.
        """
        
        response = self.llm_service.generate(prompt)
        return response.get('summary', '')