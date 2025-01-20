"""
Task 1: Exploratory Data Analysis (EDA)
This module handles basic data exploration and text analysis of Reuters headlines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from utils.helpers import clean_text, get_word_stats

class ReutersEDA:
    def __init__(self, df):
        self.df = df
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analysis"""
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%b %d %Y')
        self.df['headline_length'] = self.df['Headlines'].str.len()

    def get_basic_stats(self):
        """Get basic dataset statistics"""
        return {
            'total_articles': len(self.df),
            'unique_headlines': self.df['Headlines'].nunique(),
            'unique_descriptions': self.df['Description'].nunique(),
            'date_range': (self.df['Time'].min(), self.df['Time'].max()),
            'missing_values': self.df.isnull().sum().to_dict()
        }

    def analyze_text_patterns(self):
        """Analyze text patterns in headlines"""
        return {
            'word_frequencies': get_word_stats(self.df['Headlines']),
            'headline_length_stats': self.df['headline_length'].describe()
        }

    def get_temporal_trends(self):
        """Analyze temporal trends"""
        return self.df['Time'].value_counts().sort_index()

    def generate_wordcloud(self):
        """Generate word cloud from headlines"""
        text = ' '.join(self.df['Headlines'].apply(clean_text))
        return WordCloud(width=800, height=400, background_color='white').generate(text) 