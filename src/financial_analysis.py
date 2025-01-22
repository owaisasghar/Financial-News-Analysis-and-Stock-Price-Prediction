import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Sentiment Analysis Module
class SentimentAnalyzer:
    def __init__(self, df):
        self.df = df
        self.calculate_sentiments()

    def calculate_sentiments(self):
        """Calculate sentiment scores for headlines and descriptions using TextBlob and VADER."""
        # TextBlob for basic sentiment
        self.df['headline_sentiment'] = self.df['Headlines'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['description_sentiment'] = self.df['Description'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        # VADER for advanced sentiment
        vader_analyzer = SentimentIntensityAnalyzer()
        self.df['headline_vader'] = self.df['Headlines'].apply(
            lambda x: vader_analyzer.polarity_scores(x)['compound']
        )
        self.df['description_vader'] = self.df['Description'].apply(
            lambda x: vader_analyzer.polarity_scores(x)['compound']
        )

    def get_sentiment_stats(self):
        """Get sentiment statistics."""
        return {
            'headline_stats': self.df['headline_sentiment'].describe(),
            'description_stats': self.df['description_sentiment'].describe(),
            'headline_vader_stats': self.df['headline_vader'].describe(),
            'description_vader_stats': self.df['description_vader'].describe()
        }

    def get_sentiment_trends(self):
        """Get sentiment trends over time."""
        return self.df.groupby('Time').agg({
            'headline_sentiment': 'mean',
            'description_sentiment': 'mean',
            'headline_vader': 'mean',
            'description_vader': 'mean'
        }).reset_index()

    def get_extreme_sentiments(self, n=5):
        """Get most positive and negative headlines."""
        return {
            'most_positive': self.df.nlargest(n, 'headline_sentiment'),
            'most_negative': self.df.nsmallest(n, 'headline_sentiment')
        }

    def get_sentiment_correlation(self):
        """Calculate correlation between headline and description sentiments."""
        return self.df['headline_sentiment'].corr(self.df['description_sentiment'])


# Financial Data Retrieval Module
class FinancialAnalyzer:
    def __init__(self):
        """Initialize the financial analyzer."""
        self.cache = {}  # Cache for storing financial data

    def get_historical_data(self, ticker, start=None, end=None, period='1y', interval='1d'):
        """
        Get historical price data with error handling and caching.
        """
        try:
            stock = yf.Ticker(ticker)
            
            if start and end:
                hist_data = stock.history(
                    start=start,
                    end=end,
                    interval=interval
                )
            else:
                hist_data = stock.history(
                    period=period,
                    interval=interval
                )
            
            if hist_data.empty:
                return None, f"No historical data available for {ticker}"
            
            # Add additional calculated columns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            hist_data['Volatility'] = hist_data['Daily_Return'].rolling(window=20).std()
            hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
            hist_data['RSI'] = self.calculate_rsi(hist_data)
            
            return hist_data, None
            
        except Exception as e:
            return None, f"Error fetching historical data for {ticker}: {str(e)}"

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = prices['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# News Impact Analysis Module
class NewsImpactAnalyzer:
    def __init__(self, news_df, stock_ticker):
        """
        Initialize the news impact analyzer.
        """
        self.news_df = news_df
        self.stock_ticker = stock_ticker
        self.sentiment_analyzer = SentimentAnalyzer(news_df)
        self.financial_analyzer = FinancialAnalyzer()

    def analyze_news_impact(self, window_days=5):
        """
        Analyze the impact of news on stock performance.
        """
        # Get sentiment trends
        sentiment_trends = self.sentiment_analyzer.get_sentiment_trends()
        sentiment_trends['Time'] = pd.to_datetime(sentiment_trends['Time']).dt.tz_localize(None)
        sentiment_trends = sentiment_trends.sort_values('Time')

        # Calculate the date range for stock data
        start_date = sentiment_trends['Time'].min()
        end_date = sentiment_trends['Time'].max() + timedelta(days=window_days)

        # Get stock data for the specific period
        stock_data, error = self.financial_analyzer.get_historical_data(
            self.stock_ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        if error:
            return {'error': error}

        # Reset index and ensure Date column is timezone-naive
        stock_data = stock_data.reset_index()
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)

        # Merge sentiment and stock data
        merged_data = pd.merge_asof(
            sentiment_trends,
            stock_data,
            left_on='Time',
            right_on='Date',
            direction='nearest',
            tolerance=pd.Timedelta('1d')
        )

        # Drop rows with NaN values
        merged_data = merged_data.dropna(subset=['headline_sentiment', 'Close'])

        if len(merged_data) == 0:
            return {
                'error': 'No valid data after merging sentiment and stock data',
                'correlations': {'headline_sentiment_correlation': np.nan, 'description_sentiment_correlation': np.nan},
                'extreme_events': {
                    f'avg_return_after_positive_news_{window_days}d': np.nan,
                    f'avg_return_after_negative_news_{window_days}d': np.nan
                },
                'confidence_metrics': {
                    'overall_confidence': np.nan,
                    'sample_size_confidence': 0.0,
                    'correlation_strength_confidence': np.nan,
                    'relationship_consistency': np.nan
                }
            }

        # Calculate forward returns
        merged_data['forward_return'] = merged_data['Close'].pct_change(periods=window_days).shift(-window_days)
        merged_data = merged_data.dropna(subset=['forward_return'])

        # Calculate correlations
        correlations = {
            'headline_sentiment_correlation': stats.spearmanr(
                merged_data['headline_sentiment'],
                merged_data['forward_return']
            )[0],
            'description_sentiment_correlation': stats.spearmanr(
                merged_data['description_sentiment'],
                merged_data['forward_return']
            )[0]
        }

        # Analyze extreme sentiment events
        extreme_events = self._analyze_extreme_events(merged_data, window_days)

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(merged_data, correlations)

        return {
            'correlations': correlations,
            'extreme_events': extreme_events,
            'confidence_metrics': confidence_metrics,
            'data_points': len(merged_data)
        }

    def _analyze_extreme_events(self, merged_data, window_days):
        """Analyze the impact of extreme sentiment events on stock returns."""
        if len(merged_data) < 10:
            return {
                f'avg_return_after_positive_news_{window_days}d': np.nan,
                f'avg_return_after_negative_news_{window_days}d': np.nan
            }

        # Define extreme events (top and bottom 10% of sentiment)
        sentiment_threshold = merged_data['headline_sentiment'].quantile([0.1, 0.9])

        extreme_positive = merged_data[
            merged_data['headline_sentiment'] >= sentiment_threshold[0.9]
        ]['forward_return'].mean()

        extreme_negative = merged_data[
            merged_data['headline_sentiment'] <= sentiment_threshold[0.1]
        ]['forward_return'].mean()

        return {
            f'avg_return_after_positive_news_{window_days}d': extreme_positive,
            f'avg_return_after_negative_news_{window_days}d': extreme_negative
        }

    def _calculate_confidence_metrics(self, merged_data, correlations):
        """Calculate confidence metrics for the analysis."""
        sample_size = len(merged_data)
        sample_confidence = min(sample_size / 100, 1.0)

        if sample_size < 2:
            return {
                'overall_confidence': np.nan,
                'sample_size_confidence': sample_confidence,
                'correlation_strength_confidence': np.nan,
                'relationship_consistency': np.nan
            }

        correlation_strength = abs(correlations['headline_sentiment_correlation'])
        if np.isnan(correlation_strength):
            correlation_strength = 0.0

        # Consistency confidence (stability of relationship over time)
        try:
            rolling_corr = merged_data['headline_sentiment'].rolling(
                min(30, max(2, sample_size // 3))
            ).corr(merged_data['forward_return'])
            consistency = 1 - rolling_corr.std()
            if np.isnan(consistency):
                consistency = 0.0
        except Exception:
            consistency = 0.0

        # Combined confidence score
        confidence_score = np.mean([
            sample_confidence,
            correlation_strength,
            consistency
        ])

        return {
            'overall_confidence': confidence_score,
            'sample_size_confidence': sample_confidence,
            'correlation_strength_confidence': correlation_strength,
            'relationship_consistency': consistency
        }


# Example Usage
if __name__ == "__main__":
    # Load news data
    news_df = pd.read_csv('stock_mentions.csv')
    news_df['Time'] = pd.to_datetime(news_df['Time'])

    # Initialize analyzer
    analyzer = NewsImpactAnalyzer(news_df, 'AAPL')

    # Analyze news impact
    results = analyzer.analyze_news_impact(window_days=5)

    # Print results
    print("\nCorrelation between news and stock performance:")
    for metric, value in results['correlations'].items():
        print(f"{metric}: {value:.3f}")

    print("\nImpact of extreme sentiment events:")
    for metric, value in results['extreme_events'].items():
        print(f"{metric}: {value*100:.2f}%")

    print("\nConfidence metrics:")
    for metric, value in results['confidence_metrics'].items():
        print(f"{metric}: {value:.3f}")

    print(f"\nNumber of valid data points: {results.get('data_points', 0)}")