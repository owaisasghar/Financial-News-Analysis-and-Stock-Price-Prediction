"""
Task 2: Stock Ticker Extraction
This module handles extraction of stock tickers and company names from news articles.
"""

import re
from utils.helpers import get_stock_mapping
import pandas as pd
class StockExtractor:
    def __init__(self, df):
        self.df = df
        self.company_to_ticker = get_stock_mapping()

    def extract_stock_info(self, text):
        """Extract stock tickers and company names from text"""
        ticker_pattern = r'[\$\(]([A-Z]{1,5})[\)]?'
        tickers = re.findall(ticker_pattern, text)
        
        text_lower = text.lower()
        companies = [
            (company, ticker) 
            for company, ticker in self.company_to_ticker.items()
            if company in text_lower
        ]
        
        return list(set(tickers)), companies

    def get_all_stock_mentions(self):
        """Get all stock mentions from headlines and descriptions"""
        stock_mentions = []
        
        for _, row in self.df.iterrows():
            headline_tickers, headline_companies = self.extract_stock_info(row['Headlines'])
            desc_tickers, desc_companies = self.extract_stock_info(row['Description'])
            
            all_tickers = set(
                headline_tickers + 
                [comp[1] for comp in headline_companies] +
                desc_tickers + 
                [comp[1] for comp in desc_companies]
            )
            
            if all_tickers:
                for ticker in all_tickers:
                    stock_mentions.append({
                        'Date': row['Time'],
                        'Ticker': ticker,
                        'Headline': row['Headlines'],
                        'Description': row['Description']
                    })
        
        return pd.DataFrame(stock_mentions) 