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
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import plotly.graph_objects as go

# Sentiment Analysis Module
class SentimentAnalyzer:
    def __init__(self, df):
        self.df = df.copy()  # Create a copy of the DataFrame
        
        # Ensure Date column is datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Handle different column names
        if 'Headline' in self.df.columns and 'Headlines' not in self.df.columns:
            self.df['Headlines'] = self.df['Headline']
        
        self.calculate_sentiments()

    def calculate_sentiments(self):
        """Calculate sentiment scores for headlines and descriptions using TextBlob and VADER."""
        # TextBlob for basic sentiment
        self.df['headline_sentiment'] = self.df['Headlines'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
        )
        self.df['description_sentiment'] = self.df['Description'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
        )

        # VADER for advanced sentiment
        vader_analyzer = SentimentIntensityAnalyzer()
        self.df['headline_vader'] = self.df['Headlines'].apply(
            lambda x: vader_analyzer.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
        )
        self.df['description_vader'] = self.df['Description'].apply(
            lambda x: vader_analyzer.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
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
        # Group by date and calculate mean sentiments
        trends = self.df.groupby('Date').agg({
            'headline_sentiment': 'mean',
            'description_sentiment': 'mean',
            'headline_vader': 'mean',
            'description_vader': 'mean'
        }).reset_index()
        
        # Ensure Date is datetime
        trends['Date'] = pd.to_datetime(trends['Date'])
        return trends.sort_values('Date')

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
        self.news_df = news_df.copy()  # Create a copy of the DataFrame
        self.stock_ticker = stock_ticker
        
        # Convert Date column to datetime
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date'])
        
        # Rename columns to match expected format
        if 'Headline' in self.news_df.columns:
            self.news_df['Headlines'] = self.news_df['Headline']
        
        # Initialize sentiment analyzer and calculate sentiments
        self.sentiment_analyzer = SentimentAnalyzer(self.news_df)
        
        # Add sentiment scores to news_df
        sentiment_data = self.sentiment_analyzer.df[['headline_sentiment', 'description_sentiment', 
                                                   'headline_vader', 'description_vader']]
        self.news_df = pd.concat([self.news_df, sentiment_data], axis=1)
        
        # Initialize financial analyzer
        self.financial_analyzer = FinancialAnalyzer()

    def analyze_news_impact(self, window_days=5):
        """
        Analyze the impact of news on stock performance.
        """
        # Get sentiment trends
        sentiment_trends = self.sentiment_analyzer.get_sentiment_trends()
        sentiment_trends['Date'] = pd.to_datetime(sentiment_trends['Date']).dt.tz_localize(None)
        sentiment_trends = sentiment_trends.sort_values('Date')

        # Calculate the date range for stock data
        start_date = sentiment_trends['Date'].min()
        end_date = sentiment_trends['Date'].max() + timedelta(days=window_days)

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
            left_on='Date',
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

    def analyze_comprehensive_impact(self, window_days=5):
        """Comprehensive analysis of news impact on stock performance."""
        # Get base analysis
        base_results = self.analyze_news_impact(window_days)
        
        # Additional analyses
        keyword_impact = self._analyze_keyword_impact()
        topic_impact = self._analyze_topic_impact()
        temporal_patterns = self._analyze_temporal_patterns()
        volume_impact = self._analyze_volume_impact()
        
        return {
            'base_analysis': base_results,
            'keyword_impact': keyword_impact,
            'topic_impact': topic_impact,
            'temporal_patterns': temporal_patterns,
            'volume_impact': volume_impact,
            'summary': self._generate_summary(base_results, keyword_impact, topic_impact, 
                                           temporal_patterns, volume_impact)
        }
    
    def _analyze_keyword_impact(self):
        """Analyze impact of specific keywords on stock movement."""
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = tfidf.fit_transform(self.news_df['Description'])
        feature_names = tfidf.get_feature_names_out()
        
        # Calculate average sentiment for each keyword
        keyword_impact = defaultdict(list)
        for idx, feature in enumerate(feature_names):
            # Get articles containing this keyword
            mask = text_features[:, idx].toarray().flatten() > 0
            if mask.any():
                avg_sentiment = self.news_df.loc[mask, 'headline_sentiment'].mean()
                keyword_count = mask.sum()
                keyword_impact[feature] = {
                    'frequency': int(keyword_count),
                    'avg_sentiment': float(avg_sentiment)
                }
        
        return dict(keyword_impact)
    
    def _analyze_topic_impact(self):
        """Analyze impact of different news topics."""
        topics = {
            'financial_performance': ['earnings', 'revenue', 'profit', 'loss'],
            'market_sentiment': ['market', 'investor', 'trading', 'sentiment'],
            'company_events': ['launch', 'announcement', 'release', 'event'],
            'regulatory': ['regulation', 'compliance', 'legal', 'lawsuit'],
            'management': ['ceo', 'executive', 'management', 'leadership']
        }
        
        topic_impact = {}
        for topic, keywords in topics.items():
            topic_articles = self.news_df[
                self.news_df['Description'].str.contains('|'.join(keywords), case=False)
            ]
            if not topic_articles.empty:
                topic_impact[topic] = {
                    'article_count': len(topic_articles),
                    'avg_sentiment': float(topic_articles['headline_sentiment'].mean()),
                    'std_sentiment': float(topic_articles['headline_sentiment'].std())
                }
        
        return topic_impact
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in news impact."""
        # Date is already converted to datetime in __init__
        self.news_df['hour'] = self.news_df['Date'].dt.hour
        self.news_df['day_of_week'] = self.news_df['Date'].dt.day_name()
        
        patterns = {
            'hourly_impact': self.news_df.groupby('hour')['headline_sentiment'].agg([
                'mean', 'std', 'count'
            ]).to_dict('index'),
            'daily_impact': self.news_df.groupby('day_of_week')['headline_sentiment'].agg([
                'mean', 'std', 'count'
            ]).to_dict('index')
        }
        
        return patterns
    
    def _analyze_volume_impact(self):
        """Analyze relationship between news volume and stock movement."""
        try:
            # Convert daily_news_count index to datetime
            daily_news_count = self.news_df.groupby(self.news_df['Date'].dt.date).size()
            
            if daily_news_count.empty:
                return None
            
            # Get stock data
            start_date = self.news_df['Date'].min()
            end_date = self.news_df['Date'].max()
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            stock_data, error = self.financial_analyzer.get_historical_data(
                self.stock_ticker,
                start=start_str,
                end=end_str
            )
            
            if error or stock_data is None or stock_data.empty:
                return None
            
            # Prepare stock data
            stock_data = stock_data.reset_index()
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
            
            # Convert daily_news_count to DataFrame
            news_volume_df = pd.DataFrame(daily_news_count)
            news_volume_df.index = pd.to_datetime(news_volume_df.index)
            news_volume_df = news_volume_df.reset_index()
            news_volume_df.columns = ['Date', 'news_count']
            news_volume_df['Date'] = news_volume_df['Date'].dt.date
            
            # Merge data
            merged_data = pd.merge(
                stock_data,
                news_volume_df,
                on='Date',
                how='inner'
            )
            
            if len(merged_data) > 0:
                # Calculate correlations
                volume_correlations = {
                    'news_volume_price_corr': float(merged_data['news_count'].corr(merged_data['Close'])),
                    'news_volume_volatility_corr': float(merged_data['news_count'].corr(merged_data['Volatility'])),
                    'high_volume_days': int(len(merged_data[
                        merged_data['news_count'] > merged_data['news_count'].mean() + 
                        merged_data['news_count'].std()
                    ]))
                }
                return volume_correlations
            
            return None
            
        except Exception as e:
            print(f"Warning: Error in volume impact analysis: {str(e)}")
            return None

    def _generate_summary(self, base_results, keyword_impact, topic_impact, 
                         temporal_patterns, volume_impact):
        """Generate comprehensive summary of the analysis."""
        try:
            summary = {
                'key_findings': [],
                'risk_factors': [],
                'confidence_assessment': [],
                'recommendations': []
            }
            
            # Analyze sentiment correlation
            if 'correlations' in base_results:
                corr = base_results['correlations']['headline_sentiment_correlation']
                if not np.isnan(corr):
                    strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
                    direction = 'positive' if corr > 0 else 'negative'
                    summary['key_findings'].append(
                        f"{strength.title()} {direction} correlation ({corr:.2f}) between "
                        f"news sentiment and stock movement"
                    )
            
            # Analyze topic impact
            if topic_impact:
                try:
                    most_discussed = max(topic_impact.items(), key=lambda x: x[1]['article_count'])
                    most_positive = max(topic_impact.items(), key=lambda x: x[1]['avg_sentiment'])
                    summary['key_findings'].append(
                        f"Most discussed topic: {most_discussed[0]} "
                        f"({most_discussed[1]['article_count']} articles)"
                    )
                    summary['key_findings'].append(
                        f"Most positive topic: {most_positive[0]} "
                        f"(sentiment: {most_positive[1]['avg_sentiment']:.2f})"
                    )
                except Exception as e:
                    print(f"Warning: Error analyzing topic impact: {str(e)}")
            
            # Analyze temporal patterns
            if temporal_patterns and 'daily_impact' in temporal_patterns:
                try:
                    best_day = max(temporal_patterns['daily_impact'].items(), 
                                key=lambda x: x[1]['mean'])
                    summary['key_findings'].append(
                        f"Most positive news sentiment occurs on {best_day[0]}"
                    )
                except Exception as e:
                    print(f"Warning: Error analyzing temporal patterns: {str(e)}")
            
            # Add volume impact findings if available
            if volume_impact:
                try:
                    vol_price_corr = volume_impact.get('news_volume_price_corr', 0)
                    if abs(vol_price_corr) > 0.3:
                        summary['key_findings'].append(
                            f"{'Positive' if vol_price_corr > 0 else 'Negative'} correlation "
                            f"({vol_price_corr:.2f}) between news volume and price movement"
                        )
                    
                    high_volume_days = volume_impact.get('high_volume_days', 0)
                    if high_volume_days > 0:
                        summary['key_findings'].append(
                            f"Found {high_volume_days} days with unusually high news volume"
                        )
                except Exception as e:
                    print(f"Warning: Error analyzing volume impact: {str(e)}")
            
            # Risk assessment
            if 'confidence_metrics' in base_results:
                conf = base_results['confidence_metrics']
                if conf['overall_confidence'] < 0.5:
                    summary['risk_factors'].append(
                        "Low confidence in analysis due to limited data or inconsistent patterns"
                    )
                if conf['relationship_consistency'] < 0.3:
                    summary['risk_factors'].append(
                        "High variability in news-price relationship"
                    )
            
            # Confidence assessment
            if 'confidence_metrics' in base_results:
                conf = base_results['confidence_metrics']
                summary['confidence_assessment'] = [
                    f"Overall confidence: {conf['overall_confidence']:.2f}",
                    f"Sample size confidence: {conf['sample_size_confidence']:.2f}",
                    f"Relationship consistency: {conf['relationship_consistency']:.2f}"
                ]
            
            # Generate recommendations
            if 'confidence_metrics' in base_results:
                conf = base_results['confidence_metrics']
                if conf['overall_confidence'] > 0.7:
                    summary['recommendations'].append(
                        "High confidence in analysis - suitable for decision making"
                    )
                elif conf['overall_confidence'] > 0.4:
                    summary['recommendations'].append(
                        "Moderate confidence - use as one of multiple decision factors"
                    )
                else:
                    summary['recommendations'].append(
                        "Low confidence - gather more data before making decisions"
                    )
            
            return summary
            
        except Exception as e:
            print(f"Warning: Error generating summary: {str(e)}")
            return {
                'key_findings': ["Unable to generate complete analysis due to errors"],
                'risk_factors': ["Analysis incomplete - exercise caution"],
                'confidence_assessment': ["Analysis reliability cannot be determined"],
                'recommendations': ["Gather more data and retry analysis"]
            }

    def visualize_impact_analysis(self):
        """Create visualizations for the impact analysis."""
        figures = {}
        
        try:
            # Sentiment vs Stock Price Over Time
            sentiment_trends = self.sentiment_analyzer.get_sentiment_trends()
            
            if sentiment_trends.empty:
                return figures
            
            # Format dates for yfinance
            start_str = sentiment_trends['Date'].min().strftime('%Y-%m-%d')
            end_str = sentiment_trends['Date'].max().strftime('%Y-%m-%d')
            
            stock_data, error = self.financial_analyzer.get_historical_data(
                self.stock_ticker,
                start=start_str,
                end=end_str
            )
            
            if error or stock_data is None or stock_data.empty:
                return figures
            
            fig = go.Figure()
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=sentiment_trends['Date'],
                y=sentiment_trends['headline_sentiment'],
                name='News Sentiment',
                line=dict(color='blue')
            ))
            
            # Add stock price line
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name='Stock Price',
                yaxis='y2',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'News Sentiment vs Stock Price for {self.stock_ticker}',
                yaxis=dict(title='Sentiment Score'),
                yaxis2=dict(title='Stock Price', overlaying='y', side='right'),
                showlegend=True
            )
            
            figures['sentiment_vs_price'] = fig
            
        except Exception as e:
            print(f"Warning: Error creating visualizations: {str(e)}")
        
        return figures


# Example Usage
if __name__ == "__main__":
    try:
        # Load stock mentions data
        stock_mentions = pd.read_csv('stock_mentions.csv')
        print(f"\nLoaded stock mentions shape: {stock_mentions.shape}")
        
        if stock_mentions.empty:
            print("\nError: Empty stock mentions file")
            exit(1)
        
        # Get frequency of each ticker
        ticker_counts = stock_mentions['Ticker'].value_counts()
        print("\nTop 5 most mentioned stocks:")
        print(ticker_counts.head())
        
        # Analyze the most mentioned stock
        if not ticker_counts.empty:
            top_stock = ticker_counts.index[0]
            print(f"\nAnalyzing most mentioned stock: {top_stock}")
            
            # Filter news for the top stock
            stock_news = stock_mentions[stock_mentions['Ticker'] == top_stock]
            
            if stock_news.empty:
                print(f"\nError: No news found for {top_stock}")
                exit(1)
            
            print(f"\nFound {len(stock_news)} news items for {top_stock}")
            
            # Initialize analyzer
            analyzer = NewsImpactAnalyzer(stock_news, top_stock)
            
            # Perform comprehensive analysis
            results = analyzer.analyze_comprehensive_impact(window_days=5)
            
            if results is None:
                print("\nError: Analysis failed to produce results")
                exit(1)
            
            # Print summary
            print("\n" + "="*50)
            print("ANALYSIS SUMMARY")
            print("="*50)
            
            if 'summary' in results:
                print("\nKey Findings:")
                for finding in results['summary']['key_findings']:
                    print(f"- {finding}")
                
                print("\nRisk Factors:")
                for risk in results['summary']['risk_factors']:
                    print(f"- {risk}")
                
                print("\nConfidence Assessment:")
                for metric in results['summary']['confidence_assessment']:
                    print(f"- {metric}")
                
                print("\nRecommendations:")
                for rec in results['summary']['recommendations']:
                    print(f"- {rec}")
            else:
                print("\nError: No summary available in results")
            
            # Create visualizations
            figures = analyzer.visualize_impact_analysis()
            if figures:
                for name, fig in figures.items():
                    fig.show()
            else:
                print("\nNo visualizations could be generated")
        
        else:
            print("\nError: No stocks found in stock_mentions.csv")
            
    except FileNotFoundError:
        print("\nError: Could not find stock_mentions.csv file")
    except pd.errors.EmptyDataError:
        print("\nError: stock_mentions.csv is empty")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise