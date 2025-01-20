"""
Task 1 (Part 2): Sentiment Analysis
This module handles sentiment analysis of Reuters headlines and descriptions.
"""

from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class SentimentAnalyzer:
    def __init__(self, df):
        self.df = df
        self.calculate_sentiments()

    def calculate_sentiments(self):
        """Calculate sentiment scores for headlines and descriptions"""
        self.df['headline_sentiment'] = self.df['Headlines'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        self.df['description_sentiment'] = self.df['Description'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

    def get_sentiment_stats(self):
        """Get sentiment statistics"""
        return {
            'headline_stats': self.df['headline_sentiment'].describe(),
            'description_stats': self.df['description_sentiment'].describe()
        }

    def get_sentiment_trends(self):
        """Get sentiment trends over time"""
        return self.df.groupby('Time').agg({
            'headline_sentiment': 'mean',
            'description_sentiment': 'mean'
        }).reset_index()

    def get_extreme_sentiments(self, n=5):
        """Get most positive and negative headlines"""
        return {
            'most_positive': self.df.nlargest(n, 'headline_sentiment'),
            'most_negative': self.df.nsmallest(n, 'headline_sentiment')
        }

    def get_sentiment_correlation(self):
        """Calculate correlation between headline and description sentiments"""
        return self.df['headline_sentiment'].corr(self.df['description_sentiment']) 