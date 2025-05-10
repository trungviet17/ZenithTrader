import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.market_intelligence.tools import MarketSearchingTools, get_tools
from server.schema import AssetData
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_crawler')

class FinancialDataCrawler:
   
    
    def __init__(
        self,
        vector_db_path: str = "agent/market_intelligence/vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        rate_limit_pause: float = 1.0,
    ):
       
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rate_limit_pause = rate_limit_pause
    
        self.market_tools = MarketSearchingTools()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
 
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model="models/text-embedding-004",
        )
        
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
     
        try:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            logger.info(f"Connected to vector database at {self.vector_db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {str(e)}")
            # Create a new database if connection fails
            os.makedirs(self.vector_db_path, exist_ok=True)
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            logger.info(f"Created new vector database at {self.vector_db_path}")
    
    def _add_to_vector_store(self, documents: List[Document]):
        """Add documents to the vector store."""
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
    
    def crawl_company_overviews(self, tickers: List[str]) -> None:
       
        logger.info(f"Crawling company overviews for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing company overviews"):
            try:

                company_data = self.market_tools.ticket_overview_tool(ticker)
                
                content = f"""
                Company Overview: {company_data.asset_name} ({company_data.asset_symbol})
                Exchange: {company_data.asset_exchange}
                Type: {company_data.asset_type}
                Sector: {company_data.asset_sector}
                Industry: {company_data.asset_industry}
                
                Description:
                {company_data.asset_description}
                """
                
                # Create document with metadata
                document = Document(
                    page_content=content,
                    metadata={
                        "ticker": company_data.asset_symbol,
                        "company": company_data.asset_name,
                        "sector": company_data.asset_sector,
                        "industry": company_data.asset_industry,
                        "data_type": "company_overview",
                        "crawl_date": datetime.now().isoformat()
                    }
                )
                
                documents.append(document)
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling overview for {ticker}: {str(e)}")
   
        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_financial_reports(self, tickers: List[str]) -> None:
        
        logger.info(f"Crawling financial reports for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing financial reports"):
            try:
                # Get financial report
                financial_report = self.market_tools.financial_report_tool(ticker)
                
                # Split long reports into chunks
                chunks = self.text_splitter.split_text(financial_report)
                
                # Create document for each chunk
                for i, chunk in enumerate(chunks):
                    document = Document(
                        page_content=chunk,
                        metadata={
                            "ticker": ticker,
                            "data_type": "financial_report",
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "crawl_date": datetime.now().isoformat()
                        }
                    )
                    documents.append(document)
                
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling financial report for {ticker}: {str(e)}")
    
        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_price_data(self, tickers: List[str], days_back: int = 30) -> None:
       
        logger.info(f"Crawling price data for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing price data"):
            try:

                price_data = self.market_tools.get_price(ticker, days_back)
                
                document = Document(
                    page_content=price_data,
                    metadata={
                        "ticker": ticker,
                        "data_type": "price_data",
                        "days_back": days_back,
                        "crawl_date": datetime.now().isoformat()
                    }
                )
                documents.append(document)
                
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling price data for {ticker}: {str(e)}")

        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_sentiment_analysis(self, tickers: List[str], limit: int = 1000, hours_ago: int = 24) -> None:
        
        logger.info(f"Crawling sentiment analysis for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing sentiment analysis"):
            try:
                sentiment_results = self.market_tools.sentiment_analysis_tool(ticker, limit, hours_ago, is_has_hour_ago= False)
                
                for i, item in enumerate(sentiment_results):
                    document = Document(
                        page_content=item,
                        metadata={
                            "ticker": ticker,
                            "data_type": "sentiment_analysis",
                            "hours_ago": hours_ago,
                            "crawl_date": datetime.now().isoformat(),
                            "content_type": "header" if i == 0 else f"article_{i}",
                            "article_index": i
                        }
                    )
                    documents.append(document)
                
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling sentiment for {ticker}: {str(e)}")
        
        # Add all documents to vector store
        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_web_news(self, tickers: List[str]) -> None:
        
        logger.info(f"Crawling web news for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing web news"):
            try:
                # First get company name for better search
                try:
                    company_data = self.market_tools.ticket_overview_tool(ticker)
                    company_name = company_data.asset_name
                except:
                    company_name = ticker  # Fallback to ticker if overview fails
                
                # Build search query
                query = f"{company_name} {ticker} stock recent news financial performance"
                
                # Search the web
                search_results = self.market_tools.web_search_tool(query)
                
                # Combine all search results
                full_text = f"Web search results for {company_name} ({ticker}):\n\n"
                full_text += '\n\n'.join(search_results)
                
                document = Document(
                    page_content=full_text,
                    metadata={
                        "ticker": ticker,
                        "company": company_name,
                        "data_type": "web_news",
                        "search_query": query,
                        "crawl_date": datetime.now().isoformat()
                    }
                )
                documents.append(document)
                
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling web news for {ticker}: {str(e)}")
        
        # Add all documents to vector store
        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_news_api(self, tickers: List[str], hours_ago: int = 24) -> None:
        
        logger.info(f"Crawling news API data for {len(tickers)} tickers")
        documents = []
        
        for ticker in tqdm(tickers, desc="Processing news API data"):
            try:
                # First get company data for the news API
                company_data = self.market_tools.ticket_overview_tool(ticker)
                
                # Get news articles
                news_articles = self.market_tools.news_api_tool(company_data, hours_ago)
                
                # Combine all articles
                full_text = f"News articles for {company_data.asset_name} ({ticker}):\n\n"
                full_text += '\n\n'.join(news_articles)
                
                document = Document(
                    page_content=full_text,
                    metadata={
                        "ticker": ticker,
                        "company": company_data.asset_name,
                        "data_type": "news_api",
                        "hours_ago": hours_ago,
                        "crawl_date": datetime.now().isoformat()
                    }
                )
                documents.append(document)
                
                time.sleep(self.rate_limit_pause)
                
            except Exception as e:
                logger.error(f"Error crawling news API for {ticker}: {str(e)}")
        
        # Add all documents to vector store
        if documents:
            self._add_to_vector_store(documents)
    
    def crawl_all_data(self, tickers: List[str]) -> None:
        
        logger.info(f"Starting comprehensive data crawl for {len(tickers)} tickers")
        
        # Crawl each data type
        self.crawl_company_overviews(tickers)
        self.crawl_financial_reports(tickers)
        self.crawl_price_data(tickers)
        self.crawl_sentiment_analysis(tickers)
        self.crawl_web_news(tickers)
        self.crawl_news_api(tickers)
        
        logger.info("Completed comprehensive data crawl")
    
    def crawl_sector_data(self, sector_name: str, max_tickers: int = 10) -> None:
        
        logger.info(f"Finding and crawling data for {sector_name} sector")
        
        try:
            # Search for tickers in the sector
            query = f"top {max_tickers} stocks in {sector_name} sector ticker symbols list"
            search_results = self.market_tools.web_search_tool(query)
            
            # Extract potential ticker symbols (simple approach)
            tickers = set()
            for result in search_results:
                words = result.split()
                for word in words:
                    # Simple heuristic to detect likely ticker symbols
                    if word.isupper() and 1 < len(word) <= 5 and word.isalpha():
                        tickers.add(word)
            
            tickers = list(tickers)[:max_tickers]
            
            if tickers:
                logger.info(f"Found {len(tickers)} potential tickers for {sector_name} sector: {tickers}")
                self.crawl_all_data(tickers)
            else:
                logger.warning(f"No tickers found for {sector_name} sector")
                
        except Exception as e:
            logger.error(f"Error crawling sector data for {sector_name}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    crawler = FinancialDataCrawler(vector_db_path="agent/market_intelligence/vector_store")
    
    # Define tickers to crawl
    popular_tickers = [
        "META",  # Meta Platforms (formerly Facebook)
        "NVDA",  # NVIDIA Corporation
        "JPM",   # JPMorgan Chase
        "V",     # Visa Inc.
        "JNJ",   # Johnson & Johnson
        "WMT",   # Walmart
        "PG",    # Procter & Gamble
        "DIS",   # Walt Disney Company
        "NFLX",  # Netflix
        "KO",    # Coca-Cola Company
        "MCD",   # McDonald's
        "BA",    # Boeing
        "INTC",  # Intel
        "AMD",   # Advanced Micro Devices
        "PYPL"   # PayPal
    ]
    
    # Crawl all data for these tickers
    crawler.crawl_all_data(tickers=popular_tickers)