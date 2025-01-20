import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import spacy
from wordcloud import WordCloud
import yfinance as yf
import pandas_market_calendars as mcal
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

# Load spaCy model
@st.cache_resource
def load_spacy():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        st.error("Please install spaCy model using: python -m spacy download en_core_web_sm")
        return None

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv('reuters_headlines.csv')
    df['Time'] = pd.to_datetime(df['Time'], format='%b %d %Y')
    return df

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def get_word_stats(text_series):
    all_text = ' '.join(text_series.apply(clean_text))
    words = word_tokenize(all_text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return Counter(words)

def get_stock_mapping():
    """Returns a dictionary mapping company names to their stock tickers"""
    # Common company names to ticker mapping
    # This can be expanded or replaced with an API call
    return {
        'apple': 'AAPL',
        'tesla': 'TSLA',
        'amazon': 'AMZN',
        'microsoft': 'MSFT',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'facebook': 'META',
        'meta': 'META',
        'netflix': 'NFLX',
        'disney': 'DIS',
        'walt disney': 'DIS',
        'twitter': 'TWTR',
        'wirecard': 'WRCDF',
        'tiktok': 'BDNCE',  # ByteDance (private company)
        'nvidia': 'NVDA',
        'intel': 'INTC',
        'amd': 'AMD',
        'boeing': 'BA',
        'coca cola': 'KO',
        'coca-cola': 'KO',
        'walmart': 'WMT',
        'goldman sachs': 'GS',
        'jpmorgan': 'JPM',
        'jp morgan': 'JPM',
        'bank of america': 'BAC',
        'morgan stanley': 'MS'
    }

def extract_stock_info(text, company_to_ticker):
    """Extract stock tickers and company names from text"""
    # Direct ticker pattern (e.g., $AAPL or (AAPL))
    ticker_pattern = r'[\$\(]([A-Z]{1,5})[\)]?'
    tickers = re.findall(ticker_pattern, text)
    
    # Look for company names
    text_lower = text.lower()
    companies = []
    for company, ticker in company_to_ticker.items():
        if company in text_lower:
            companies.append((company, ticker))
    
    return list(set(tickers)), companies

def validate_ticker(ticker):
    """Validate if a ticker exists using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return 'symbol' in info
    except:
        return False

def get_financial_metrics(ticker):
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
        
        # Format market cap to billions
        if isinstance(metrics['Market Cap'], (int, float)):
            metrics['Market Cap'] = f"${metrics['Market Cap']/1e9:.2f}B"
            
        # Format dividend yield to percentage
        if isinstance(metrics['Dividend Yield'], float):
            metrics['Dividend Yield'] = f"{metrics['Dividend Yield']*100:.2f}%"
            
        return metrics, None
    except Exception as e:
        return None, str(e)

def get_historical_data(ticker, period='1y'):
    """Get historical price data for a stock"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist, None
    except Exception as e:
        return None, str(e)

def calculate_returns(prices):
    """Calculate various return metrics"""
    if len(prices) < 2:
        return None
    
    daily_returns = prices['Close'].pct_change()
    
    metrics = {
        'Daily Avg Return': daily_returns.mean(),
        'Daily Std Dev': daily_returns.std(),
        'Annualized Return': daily_returns.mean() * 252,
        'Annualized Volatility': daily_returns.std() * np.sqrt(252),
        'Total Return': (prices['Close'].iloc[-1] / prices['Close'].iloc[0]) - 1
    }
    
    return metrics

def main():
    st.title("Reuters Headlines Analysis Dashboard")
    st.write("Analysis of Reuters news headlines from July 2020")

    # Initialize resources
    download_nltk_data()
    nlp = load_spacy()
    df = load_data()

    # Sidebar navigation with task-based organization
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Task",
        [
            "Task 1: Exploratory Data Analysis (EDA)",
            "Task 2: Stock Ticker Extraction",
            "Task 3: Financial Data Retrieval",
        ]
    )

    if analysis_type == "Task 1: Exploratory Data Analysis (EDA)":
        st.header("Task 1: Exploratory Data Analysis (EDA)")
        
        # Add EDA subtabs
        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "Dataset Overview", 
            "Text Analysis", 
            "Temporal Trends",
            "Sentiment Analysis"
        ])
        
        with eda_tab1:
            st.subheader("1.1 Basic Dataset Properties")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Articles", len(df))
                st.metric("Unique Headlines", df['Headlines'].nunique())
            with col2:
                st.metric("Date Range", f"{df['Time'].min().date()} to {df['Time'].max().date()}")
                st.metric("Unique Descriptions", df['Description'].nunique())

            st.subheader("1.2 Data Quality Check")
            st.write("Missing Values Analysis:")
            st.write(df.isnull().sum())
            
            st.subheader("1.3 Sample Data")
            st.dataframe(df.head())

        with eda_tab2:
            st.subheader("1.4 Text Pattern Analysis")
            headline_word_freq = get_word_stats(df['Headlines'])
            
            st.write("Most Common Words in Headlines")
            freq_df = pd.DataFrame(headline_word_freq.most_common(10), columns=['Word', 'Count'])
            st.bar_chart(freq_df.set_index('Word'))

            st.write("Headline Length Distribution")
            df['headline_length'] = df['Headlines'].str.len()
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='headline_length', bins=20)
            st.pyplot(fig)

        with eda_tab3:
            st.subheader("1.5 Temporal Analysis")
            st.write("Articles Published per Day")
            daily_counts = df['Time'].value_counts().sort_index()
            st.line_chart(daily_counts)

            # Word cloud visualization
            st.write("Word Cloud of Headlines")
            words = ' '.join(df['Headlines'].apply(clean_text))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        with eda_tab4:
            st.subheader("1.6 Sentiment Analysis")
            
            # Calculate sentiments if not already done
            if 'headline_sentiment' not in df.columns:
                df['headline_sentiment'] = df['Headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)
                df['description_sentiment'] = df['Description'].apply(lambda x: TextBlob(x).sentiment.polarity)

            # 1. Overall Sentiment Distribution
            st.write("#### 1.6.1 Overall Sentiment Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Headlines Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df['headline_sentiment'], bins=20)
                plt.title("Distribution of Headline Sentiments")
                plt.xlabel("Sentiment Score (-1 to 1)")
                st.pyplot(fig)

            with col2:
                st.write("Descriptions Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df['description_sentiment'], bins=20)
                plt.title("Distribution of Description Sentiments")
                plt.xlabel("Sentiment Score (-1 to 1)")
                st.pyplot(fig)

            # 2. Sentiment Statistics
            st.write("#### 1.6.2 Sentiment Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Headline Sentiment Statistics")
                headline_stats = df['headline_sentiment'].describe()
                st.write(pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{headline_stats['mean']:.3f}",
                        f"{headline_stats['50%']:.3f}",
                        f"{headline_stats['std']:.3f}",
                        f"{headline_stats['min']:.3f}",
                        f"{headline_stats['max']:.3f}"
                    ]
                }))

            with col2:
                st.write("Description Sentiment Statistics")
                desc_stats = df['description_sentiment'].describe()
                st.write(pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{desc_stats['mean']:.3f}",
                        f"{desc_stats['50%']:.3f}",
                        f"{desc_stats['std']:.3f}",
                        f"{desc_stats['min']:.3f}",
                        f"{desc_stats['max']:.3f}"
                    ]
                }))

            # 3. Sentiment Over Time
            st.write("#### 1.6.3 Sentiment Trends Over Time")
            daily_sentiment = df.groupby('Time').agg({
                'headline_sentiment': 'mean',
                'description_sentiment': 'mean'
            }).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_sentiment['Time'],
                y=daily_sentiment['headline_sentiment'],
                name='Headlines'
            ))
            fig.add_trace(go.Scatter(
                x=daily_sentiment['Time'],
                y=daily_sentiment['description_sentiment'],
                name='Descriptions'
            ))
            fig.update_layout(
                title='Average Daily Sentiment',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                showlegend=True
            )
            st.plotly_chart(fig)

            # 4. Extreme Sentiment Analysis
            st.write("#### 1.6.4 Extreme Sentiment Analysis")
            
            # Most positive and negative headlines
            st.write("Most Positive Headlines:")
            most_positive = df.nlargest(5, 'headline_sentiment')[['Headlines', 'headline_sentiment']]
            st.dataframe(most_positive)

            st.write("Most Negative Headlines:")
            most_negative = df.nsmallest(5, 'headline_sentiment')[['Headlines', 'headline_sentiment']]
            st.dataframe(most_negative)

            # 5. Sentiment Category Distribution
            st.write("#### 1.6.5 Sentiment Category Distribution")
            
            def get_sentiment_category(score):
                if score <= -0.1:
                    return 'Negative'
                elif score >= 0.1:
                    return 'Positive'
                else:
                    return 'Neutral'

            df['headline_sentiment_category'] = df['headline_sentiment'].apply(get_sentiment_category)
            sentiment_dist = df['headline_sentiment_category'].value_counts()

            fig, ax = plt.subplots()
            plt.pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%')
            plt.title('Distribution of Sentiment Categories')
            st.pyplot(fig)

            # 6. Correlation Analysis
            st.write("#### 1.6.6 Headline vs Description Sentiment Correlation")
            correlation = df['headline_sentiment'].corr(df['description_sentiment'])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")

            fig = px.scatter(df, 
                            x='headline_sentiment', 
                            y='description_sentiment',
                            title='Headline vs Description Sentiment Correlation')
            st.plotly_chart(fig)

    elif analysis_type == "Task 2: Stock Ticker Extraction":
        st.header("Task 2: Stock Ticker Extraction")
        
        # Initialize stock mapping
        company_to_ticker = get_stock_mapping()
        
        # Create a DataFrame to store stock mentions
        stock_mentions = []
        
        with st.spinner('Extracting stock information...'):
            for idx, row in df.iterrows():
                headline_tickers, headline_companies = extract_stock_info(row['Headlines'], company_to_ticker)
                desc_tickers, desc_companies = extract_stock_info(row['Description'], company_to_ticker)
                
                all_tickers = set(headline_tickers + [comp[1] for comp in headline_companies] +
                                desc_tickers + [comp[1] for comp in desc_companies])
                
                if all_tickers:
                    for ticker in all_tickers:
                        stock_mentions.append({
                            'Date': row['Time'],
                            'Ticker': ticker,
                            'Headline': row['Headlines'],
                            'Description': row['Description']
                        })
        
        if stock_mentions:
            mentions_df = pd.DataFrame(stock_mentions)
            
            st.subheader("2.1 Stock Mentions Overview")
            ticker_counts = mentions_df['Ticker'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Stock Mentions", len(mentions_df))
                st.metric("Unique Stocks Mentioned", len(ticker_counts))
            
            with col2:
                st.write("Most Mentioned Stocks")
                st.bar_chart(ticker_counts.head(10))
            
            st.subheader("2.2 Detailed Stock Mentions")
            st.dataframe(mentions_df.sort_values('Date', ascending=False))
            
            # Download option
            csv = mentions_df.to_csv(index=False)
            st.download_button(
                label="Download Stock Mentions Data",
                data=csv,
                file_name="stock_mentions.csv",
                mime="text/csv"
            )

    elif analysis_type == "Task 3: Financial Data Retrieval":
        st.header("Task 3: Financial Data Retrieval")
        
        # Get stock mentions first
        company_to_ticker = get_stock_mapping()
        stock_mentions = []
        
        for idx, row in df.iterrows():
            headline_tickers, headline_companies = extract_stock_info(row['Headlines'], company_to_ticker)
            desc_tickers, desc_companies = extract_stock_info(row['Description'], company_to_ticker)
            all_tickers = set(headline_tickers + [comp[1] for comp in headline_companies] +
                            desc_tickers + [comp[1] for comp in desc_companies])
            if all_tickers:
                for ticker in all_tickers:
                    stock_mentions.append(ticker)
        
        unique_tickers = list(set(stock_mentions))
        
        if unique_tickers:
            fin_tab1, fin_tab2 = st.tabs(["Financial Metrics", "Historical Data"])
            
            with fin_tab1:
                st.subheader("3.1 Key Financial Metrics")
                selected_ticker = st.selectbox(
                    "Select a stock for analysis",
                    options=sorted(unique_tickers)
                )
                
                if selected_ticker:
                    metrics, error = get_financial_metrics(selected_ticker)
                    if metrics:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Market Cap", metrics['Market Cap'])
                            st.metric("P/E Ratio", metrics['P/E Ratio'])
                            st.metric("Forward P/E", metrics['Forward P/E'])
                            st.metric("PEG Ratio", metrics['PEG Ratio'])
                            st.metric("Price to Book", metrics['Price to Book'])
                        with col2:
                            st.metric("Dividend Yield", metrics['Dividend Yield'])
                            st.metric("52 Week High", metrics['52 Week High'])
                            st.metric("52 Week Low", metrics['52 Week Low'])
                            st.metric("Beta", metrics['Beta'])
                            st.metric("Industry", metrics['Industry'])
                    else:
                        st.error(f"Error fetching metrics: {error}")
            
            with fin_tab2:
                st.subheader("3.2 Historical Price Analysis")
                selected_ticker = st.selectbox(
                    "Select a stock",
                    options=sorted(unique_tickers),
                    key='hist_ticker'
                )
                
                period = st.selectbox(
                    "Select time period",
                    options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                    index=3
                )
                
                if selected_ticker:
                    hist_data, error = get_historical_data(selected_ticker, period)
                    if hist_data is not None:
                        # Price and volume chart
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          vertical_spacing=0.03,
                                          row_heights=[0.7, 0.3])
                        
                        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'],
                                               name='Close Price'),
                                    row=1, col=1)
                        
                        fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'],
                                           name='Volume'),
                                    row=2, col=1)
                        
                        fig.update_layout(height=600, title_text=f"{selected_ticker} Price and Volume")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Return metrics
                        returns = calculate_returns(hist_data)
                        if returns:
                            st.subheader("3.3 Return Metrics")
                            metrics_df = pd.DataFrame({
                                'Metric': returns.keys(),
                                'Value': [f"{v*100:.2f}%" for v in returns.values()]
                            })
                            st.table(metrics_df)
                    else:
                        st.error(f"Error fetching historical data: {error}")
        else:
            st.warning("No stock mentions found in the dataset")

    # Add task descriptions in sidebar
    st.sidebar.markdown("""
    ### Task Descriptions
    
    **Task 1: EDA**
    - Basic dataset properties
    - Text patterns analysis
    - Temporal trends
    - Sentiment analysis
    
    **Task 2: Stock Extraction**
    - Company name identification
    - Stock ticker mapping
    - Mention frequency analysis
    
    **Task 3: Financial Data**
    - Key financial metrics
    - Historical price data
    - Return analysis
    """)

if __name__ == "__main__":
    main() 