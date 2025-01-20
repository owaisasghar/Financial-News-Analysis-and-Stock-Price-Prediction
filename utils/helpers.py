"""
Helper functions used across different modules
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def clean_text(text):
    """Clean text by removing special characters and converting to lowercase"""
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

def get_word_stats(text_series):
    """Get word frequency statistics from text series"""
    all_text = ' '.join(text_series.apply(clean_text))
    words = word_tokenize(all_text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(words)

def get_stock_mapping():
    """Returns mapping of company names to stock tickers"""
    return {
        'apple': 'AAPL',
        'tesla': 'TSLA',
        # ... (rest of the mapping)
    } 