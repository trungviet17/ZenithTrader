import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from typing import List, Optional
from langchain.tools import BaseTool, Tool
from datetime import datetime, timedelta
from langchain_community.tools import TavilySearchResults
from server.schema import AssetData
from newsapi import NewsApiClient
import requests 

from dotenv import load_dotenv
import os 
load_dotenv() 


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_AVANTAGE_API_KEY = os.getenv("ALPHA_AVANTAGE_API_KEY")




class MarketSearchingTools: 

    def __init__(self, 
                 tavily_api_key: str = TAVILY_API_KEY,
                 news_api_key: str = NEWS_API_KEY,
                 alpha_avantage_api_key: str = ALPHA_AVANTAGE_API_KEY,
                 ) -> None: 

        self.tavily_api_key = tavily_api_key
        self.news_api_key = news_api_key
        self.alpha_avantage_api_key = alpha_avantage_api_key


    def web_search_tool(self, query : str) -> str:
        """
        This tool performs a web search and returns the results.
        Args:
            query: The search query
        Returns:
            Formatted search results
        """
        tavily_search = TavilySearchResults( 
            max_results = 2, 
            api_key = self.tavily_api_key,
        )
        results = tavily_search.invoke(query)

        formatted_information = "Search results: \n\n"
        for i, result in enumerate(results):
            formatted_information += f"Result {i+1}:\n"
            formatted_information += f"Title: {result['title']}\n"
            formatted_information += f"Description: {result['content']}\n\n"


        return formatted_information

    
    def news_api_tool(self, data: AssetData) -> str:
        """
        This tool performs a news API search and returns the results.
        Args:
            data: Asset data containing name and other information
        Returns:
            Formatted news results
        """

        client = NewsApiClient( api_key  = self.news_api_key)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        all_articles = client.get_everything(
            q = data.asset_name,
            from_param = one_hour_ago.strftime("%Y-%m-%dT%H:%M:%S"),
            to = datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            language = "en",
            sort_by = "relevancy",
            page_size = 5,
        )

        formatted_information = "News results: \n\n"

        for i, article in enumerate(all_articles["articles"]):
            formatted_information += f"Article {i+1}:\n"
            formatted_information += f"Title: {article['title']}\n"
            formatted_information += f"Description: {article['description']}\n\n"


        return formatted_information

    def financial_report_tool(self, ticker: str) -> str:
        """
        Retrieves and formats comprehensive financial reports for a given stock ticker.
        This function fetches both income statement and balance sheet data for the specified company,
        then analyzes and presents the data in a structured format with key financial metrics and ratios.
        
        Args:
            ticker: The stock ticker symbol of the company (e.g., 'AAPL', 'MSFT', 'GOOGL')
            
        Returns:
            A formatted string containing the financial report
        """
        # take income statement 
        income_url = "https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=" + str(ticker) + "&apikey=" + self.alpha_avantage_api_key
        balance_url = "https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=" + str(ticker) + "&apikey=" + self.alpha_avantage_api_key

        response_income = requests.get(income_url)
        response_balance = requests.get(balance_url)
        
        if response_income.status_code != 200 or response_balance.status_code != 200:
            raise Exception(f"API fetch error. Status: Income={response_income.status_code}, Balance={response_balance.status_code}")

        income_data = response_income.json()
        balance_data = response_balance.json()

        # take data by quarter and annual 
        income_annual = income_data['annualReports'][0]
        income_quarter = income_data['quarterlyReports'][0]

        balance_annual = balance_data['annualReports'][0]
        balance_quarter = balance_data['quarterlyReports'][0]
        
        formatted_information = f"""
        Financial Summary for fiscal year ending {income_annual['fiscalDateEnding']}:

            Liquidity:
            - Cash and short-term investments: ${float(balance_annual['cashAndShortTermInvestments']) / 1e9:.2f}B
            - Total current assets: ${float(balance_annual['totalCurrentAssets']) / 1e9:.2f}B
            - Total current liabilities: ${float(balance_annual['totalCurrentLiabilities']) / 1e9:.2f}B
            - Current ratio: {(float(balance_annual['totalCurrentAssets']) / float(balance_annual['totalCurrentLiabilities'])):.2f}

            Financial Leverage:
            - Total liabilities: ${float(balance_annual['totalLiabilities']) / 1e9:.2f}B
            - Total assets: ${float(balance_annual['totalAssets']) / 1e9:.2f}B
            - Debt ratio (Total Liabilities / Total Assets): {(float(balance_annual['totalLiabilities']) / float(balance_annual['totalAssets'])):.2f}

            Debt Load:
            - Short-term debt: ${float(balance_annual['shortTermDebt']) / 1e9:.2f}B
            - Long-term debt: ${float(balance_annual['longTermDebt']) / 1e9:.2f}B
            - Total debt: ${(float(balance_annual['shortTermDebt']) + float(balance_annual['longTermDebt'])) / 1e9:.2f}B
            - EBITDA: ${float(income_annual['ebitda']) / 1e9:.2f}B
            - Debt / EBITDA: {((float(balance_annual['shortTermDebt']) + float(balance_annual['longTermDebt'])) / float(income_annual['ebitda'])):.2f} → Lower is better

            Shareholder Equity:
            - Total shareholder equity: ${float(balance_annual['totalShareholderEquity']) / 1e9:.2f}B
            - Outstanding shares: {float(balance_annual['commonStockSharesOutstanding']) / 1e6:.2f}M
            - Book Value Per Share (BVPS): ${(float(balance_annual['totalShareholderEquity']) / float(balance_annual['commonStockSharesOutstanding'])):.2f}

            Income Statement Highlights:
            - Total revenue: ${float(income_annual['totalRevenue']) / 1e9:.2f}B
            - Gross profit: ${float(income_annual['grossProfit']) / 1e9:.2f}B
            - Gross profit margin: {(float(income_annual['grossProfit']) / float(income_annual['totalRevenue']) * 100):.2f}%
            - Operating income (EBIT): ${float(income_annual['operatingIncome']) / 1e9:.2f}B
            - EBITDA: ${float(income_annual['ebitda']) / 1e9:.2f}B
            - Net income: ${float(income_annual['netIncome']) / 1e9:.2f}B

            R&D Spending (Innovation):
            - Research & Development expense: ${float(income_annual['researchAndDevelopment']) / 1e9:.2f}B

            Interpretation Tips:
            - A current ratio above 1.0 indicates healthy short-term liquidity.
            - A debt ratio below 0.6 is typically considered low-risk.
            - BVPS can be compared to market price to assess undervaluation.
            - Strong gross margin and operating income suggest good core profitability.

        Quarterly Snapshot (Quarter Ending {income_quarter['fiscalDateEnding']}):

            Revenue: ${float(income_quarter['totalRevenue']) / 1e9:.2f}B
            Net Income: ${float(income_quarter['netIncome']) / 1e9:.2f}B
            Gross Margin: {(float(income_quarter['grossProfit']) / float(income_quarter['totalRevenue']) * 100):.2f}%
            Operating Income (EBIT): ${float(income_quarter['operatingIncome']) / 1e9:.2f}B
            Short-term Debt: ${float(balance_quarter['shortTermDebt']) / 1e9:.2f}B
            Cash: ${float(balance_quarter['cashAndShortTermInvestments']) / 1e9:.2f}B
            R&D: ${float(income_quarter['researchAndDevelopment']) / 1e9:.2f}B

            Commentary:
            - Compare quarterly net income and revenue trends with annual data to assess momentum.
            - Look for seasonal effects, turnarounds, or deteriorations in margin or debt ratios.
        """
        return formatted_information


    def sentiment_analysis_tool(self, ticker: str, limit : int = 10) -> str:
        """
        Performs sentiment analysis on recent news articles related to a specified ticker symbol.
        Args:
            ticker: The stock ticker symbol to analyze (e.g., 'AAPL', 'MSFT')
            limit: Maximum number of articles to retrieve. Defaults to 10.
        Returns:
            Formatted string containing sentiment analysis results
        """
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        time_from = one_hour_ago.strftime("%Y%m%dT%H%M")


        url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=" +  ticker + "&apikey=" + self.alpha_avantage_api_key + "&limit=" + str(limit) + "&time_from=" + str(time_from) + "&sort=LATEST"

        response = requests.get(url)
        if response.status_code != 200: 
            raise Exception(f"API fetch error. Status: {response.status_code}")

        data = response.json()
        formatted_information = f"""

        Sentiment Analysis for {ticker} from {one_hour_ago.strftime("%Y-%m-%d %H:%M")} to {now.strftime("%Y-%m-%d %H:%M")}:
        - Sentiment score definition : x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish
        """

        for i, article in enumerate(data['feed']):
            formatted_information += f"""
            Article {i+1}:
            - Title: {article['title']}
            - Time Published: {article['time_published']}
            - Sentiment Score: {article['overall_sentiment_score']}
            - Sentiment Label: {article['overall_sentiment_label']}
            - Summary: {article['summary']}
            """
        
        return formatted_information

    def ticket_overview_tool(self, ticket: str) -> AssetData:
        """
        This tool retrieves comprehensive overview information about a stock ticker.
        Args:
            ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT') 
        Returns:
            AssetData object containing company information
        """

        url = "https://www.alphavantage.co/query?function=OVERVIEW&symbol=" + ticket + "&apikey=" + self.alpha_avantage_api_key
        response = requests.get(url)

        if response.status_code != 200: 
            raise Exception(f"API fetch error. Status: {response.status_code}")
        
        data = response.json()

        return AssetData(
            asset_symbol = data['Symbol'],
            asset_name  = data['Name'],
            asset_type = data['AssetType'],
            asset_exchange = data['Exchange'],
            asset_sector = data['Sector'],
            asset_industry = data['Industry'],
            asset_description = data['Description'],
        )


def get_tools()   -> List[BaseTool]:

    tools_inst = MarketSearchingTools()

    return [

        Tool(
            name="web_search",
            func=tools_inst.web_search_tool,
            description="Searches the web for information about markets, companies, or financial topics."
        ),
        Tool(
            name="news_search",
            func=tools_inst.news_api_tool,
            description="Retrieves recent news articles about a specific asset or company."
        ),
        Tool(
            name="financial_report",
            func=tools_inst.financial_report_tool,
            description="Retrieves comprehensive financial reports including income statements and balance sheets for a given ticker symbol."
        ),
        Tool(
            name="sentiment_analysis",
            func=tools_inst.sentiment_analysis_tool,
            description="Analyzes market sentiment for a specific ticker based on recent news and social media."
        ),
        Tool(
            name="ticker_overview",
            func=tools_inst.ticket_overview_tool,
            description="Gets comprehensive overview information about a company including its sector, industry, and business description."
        )

    ]





if __name__ == '__main__': 

    print("Testing tools...")
    tools = get_tools()


    def test_tavily_search_tool(query):
        try : 
            print("Testing Tavily Search Tool...")
            print("Query:", query)


            # Using the recommended invoke method:
            result = tools[0].run(query)
            print("Result:", result)
            print("Success")

        except Exception as e:
            print("Error:", e)
        


    def test_news_api_tool(data):
        try : 
            print("Testing News API Tool...")
            print("Data:", data)
            result = tools[1].run(data)
            print("Result:", result)
            print("Success")

        except Exception as e:
            print("Error:", e)


    def test_financial_report_tool(ticker):
        try : 
            print("Testing Financial Report Tool...")
            print("Ticker:", ticker)
            result = tools.financial_report_tool(ticker)
            print("Result:", result)
            print("Success")

        except Exception as e:
            print("Error:", e)

    

    def test_sentiment_analysis_tool(ticker):
        try : 
            print("Testing Sentiment Analysis Tool...")
            print("Ticker:", ticker)
            result = tools.sentiment_analysis_tool(ticker)
            print("Result:", result)
            print("Success")

        except Exception as e:
            print("Error:", e)

    def test_ticket_overview_tool(ticker):

        try : 
            print("Testing Ticket Overview Tool...")
            print("Ticker:", ticker)
            result = tools.ticket_overview_tool(ticker)
            print("Result:", result)
            print("Success")

        except Exception as e:
            print("Error:", e)



    # Test the tools
    # query = "latest news on Apple Inc"
    # test_tavily_search_tool(query)

    asset_data = AssetData(
        asset_symbol = "AAPL",
        asset_name = "Apple Inc.",
        asset_type = "stock",
        asset_exchange = "NASDAQ",
        asset_sector = "Technology",
        asset_industry = "Consumer Electronics",
        asset_description = "Apple Inc. designs, manufactures, and markets consumer electronics, software, and services."
    )


    test_news_api_tool(asset_data)
    # test_financial_report_tool("AAPL", tools)
    # test_sentiment_analysis_tool("AAPL", tools)
    # test_ticket_overview_tool("AAPL", tools)


    # print("All tests completed.")
    

    



    
    




