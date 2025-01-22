# Financial News Analysis Project

## Project Overview
This project analyzes Reuters financial news headlines to extract insights about companies, perform sentiment analysis, and analyze financial data. The project is structured into multiple tasks, each focusing on different aspects of financial news and market analysis.

## Tasks

### Task 1: News Headlines Analysis
📊 **Purpose**: Comprehensive analysis of Reuters headlines dataset
- Text pattern analysis and word frequencies
- Temporal trend visualization
- Word cloud generation
- Basic statistics and visualizations

### Task 2: Stock Extraction and Analysis
💹 **Purpose**: Identify company mentions and analyze stock data
- Company name detection in headlines
- Stock ticker mapping and validation
- Mention frequency analysis
- Stock data correlation with news

### Task 3: Financial Analysis
📈 **Purpose**: Detailed financial data analysis
- Historical price data analysis
- Key financial metrics computation
- Performance visualization
- Market trend analysis

### Task 4: Time Series Forecasting
📊 **Purpose**: Predict stock movements
- Time series analysis of stock prices
- Forecasting models implementation
- Performance evaluation
- Trend prediction

### Task 5: News Impact Analysis
📰 **Purpose**: Analyze news sentiment impact
- Sentiment analysis of headlines
- News impact on stock prices
- Correlation studies
- Market reaction analysis

## Project Structure
```
project/
├── src/
│   ├── __init__.py
│   ├── stock_extraction.py
│   ├── financial_analysis.py
│   ├── sentiment_analysis.py
│   └── news_impact_analyzer.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── Task1_Analysis.ipynb
│   ├── Task2_Stock_Extraction.ipynb
│   ├── Task3_Financial_Analysis.ipynb
│   ├── Task4-TimeSeries_Forecasting.ipynb
│   └── Task5_News_Impact.ipynb
├── results/
│   └── task_1/
│       ├── headlines_per_day.png
│       ├── headline_word_freq.png
│       └── headline_length_dist.png
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/financial-news-analysis.git
cd financial-news-analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebooks
Each task is implemented as a separate Jupyter notebook in the `notebooks/` directory. Open and run them sequentially:

```bash
jupyter notebook notebooks/
```

### Running the Streamlit App
The project includes a Streamlit web application for interactive analysis:

```bash
streamlit run app/streamlit_app.py
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Reuters for the financial news dataset
- Yahoo Finance for financial data
- Libraries: pandas, numpy, scikit-learn, NLTK, spaCy, Streamlit
```

