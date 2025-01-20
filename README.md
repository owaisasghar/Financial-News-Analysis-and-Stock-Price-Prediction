 Reuters Financial News Analysis

## Project Overview
This project analyzes Reuters financial news headlines to extract insights about companies, perform sentiment analysis, and retrieve financial data. The analysis is structured into three main tasks, each focusing on different aspects of financial news analysis.

## Tasks

### Task 1: Exploratory Data Analysis (EDA)
📊 **Purpose**: Comprehensive analysis of Reuters headlines dataset

**Features**:
- Dataset statistics and quality assessment
- Text pattern analysis and word frequencies
- Temporal trend visualization
- Word cloud generation
- Sentiment analysis of headlines and descriptions

**Key Components** (`src/eda.py`):
- `ReutersEDA` class: Handles data preparation, analysis, and visualization
- `prepare_data()`: Prepares data for analysis
- `analyze_data()`: Performs EDA tasks
- `visualize_data()`: Generates visualizations

### Task 2: Stock Ticker Extraction
💹 **Purpose**: Identify and map company mentions to stock tickers

**Features**:
- Automatic company name detection
- Stock ticker mapping
- Mention frequency analysis
- News article association

**Key Components** (`src/stock_extraction.py`):
- `extract_tickers()`: Extracts stock tickers from headlines
- `validate_tickers()`: Validates extracted tickers

### Task 3: Financial Data Retrieval
📈 **Purpose**: Analyze financial data for mentioned stocks

**Features**:
- Historical price data retrieval
- Key financial metrics analysis
- Return calculations
- Performance visualization

**Key Components** (`src/financial_analysis.py`):
```python
class FinancialAnalyzer:
    def get_financial_metrics() # Get key stock metrics
    def get_historical_data()   # Retrieve price history
    def calculate_returns()     # Compute return metrics
```

## Project Structure

```
reuters-financial-analysis/
├── src/
│   ├── eda.py                 # EDA implementation
│   ├── sentiment_analysis.py  # Sentiment analysis
│   ├── stock_extraction.py    # Stock ticker extraction
│   └── financial_analysis.py  # Financial data analysis
├── utils/
│   └── helpers.py            # Helper functions
└── data/
    └── reuters_headlines.csv # Dataset
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/reuters-financial-analysis.git
cd reuters-financial-analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage Examples

### 1. Exploratory Data Analysis
```python
from src.eda import ReutersEDA

# Initialize analyzer
eda = ReutersEDA(df)

# Get basic statistics
stats = eda.get_basic_stats()

# Generate word cloud
wordcloud = eda.generate_wordcloud()
```

### 2. Stock Ticker Extraction
```python
from src.stock_extraction import StockExtractor

# Initialize extractor
extractor = StockExtractor(df)

# Get all stock mentions
mentions = extractor.get_all_stock_mentions()
```

### 3. Financial Analysis
```python
from src.financial_analysis import FinancialAnalyzer

# Initialize analyzer
analyzer = FinancialAnalyzer()

# Get stock metrics
metrics, _ = analyzer.get_financial_metrics('AAPL')

# Get historical data
hist_data, _ = analyzer.get_historical_data('AAPL', '1y')
```

## Dependencies
```
pandas==1.5.3
streamlit==1.22.0
nltk==3.8.1
textblob==0.17.1
yfinance==0.2.18
plotly==5.14.1
wordcloud==1.9.2
spacy==3.5.2
seaborn==0.12.2
matplotlib==3.7.1
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Reuters for the financial news dataset
- Yahoo Finance for financial data
- NLTK and spaCy for NLP capabilities
```

