"""
Task 3: Financial Data Retrieval
This module handles retrieval and analysis of financial data for extracted stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np

class FinancialAnalyzer:
    def __init__(self):
        pass

    def get_financial_metrics(self, ticker):
        """Get key financial metrics for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Forward P/E': info.get('forwardPE', 'N/A'),
                'PEG Ratio': info.get('pegRatio', 'N/A'),
                'Price to Book': info.get('priceToBook', 'N/A'),
                'Dividend Yield': info.get('dividendYield', 'N/A'),
                '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                'Industry': info.get('industry', 'N/A')
            }
            
            self._format_metrics(metrics)
            return metrics, None
        except Exception as e:
            return None, str(e)

    def _format_metrics(self, metrics):
        """Format metrics for display"""
        if isinstance(metrics['Market Cap'], (int, float)):
            metrics['Market Cap'] = f"${metrics['Market Cap']/1e9:.2f}B"
        if isinstance(metrics['Dividend Yield'], float):
            metrics['Dividend Yield'] = f"{metrics['Dividend Yield']*100:.2f}%"

    def get_historical_data(self, ticker, period='1y'):
        """Get historical price data"""
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period), None
        except Exception as e:
            return None, str(e)

    def calculate_returns(self, prices):
        """Calculate return metrics"""
        if len(prices) < 2:
            return None
        
        daily_returns = prices['Close'].pct_change()
        
        return {
            'Daily Avg Return': daily_returns.mean(),
            'Daily Std Dev': daily_returns.std(),
            'Annualized Return': daily_returns.mean() * 252,
            'Annualized Volatility': daily_returns.std() * np.sqrt(252),
            'Total Return': (prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1
        } 