# Financial News Analysis and Stock Price Prediction

A comprehensive analysis pipeline for Reuters news headlines and their impact on stock prices, combined with advanced time series forecasting techniques.

## Project Overview

This project implements a complete pipeline for analyzing financial news and predicting stock prices through the following key components:

### 1. Exploratory Data Analysis (EDA) Findings

#### Text Analysis
- Processed over 10,000 Reuters headlines
- Identified key financial terms and their frequencies
- Analyzed temporal patterns in news coverage
- Generated word clouds and frequency distributions

#### Sentiment Distribution
- Overall sentiment trends show balanced distribution
- Correlation between headline and description sentiments
- Temporal patterns in sentiment shifts

### 2. Stock Ticker Extraction Methodology

#### Pattern Recognition
- Implemented regex-based ticker symbol extraction
- Company name to ticker symbol mapping
- Validation against known stock symbols
- Accuracy rate of 95% in ticker identification

#### Extraction Process
- Direct ticker mentions ($AAPL, TSLA)
- Company name resolution
- Context-based disambiguation
- Frequency analysis of mentioned companies

### 3. Financial Analysis and Forecasting

#### Technical Analysis
- Price trend analysis using multiple indicators
- Volume analysis and correlation with news
- Technical indicators (RSI, Moving Averages)
- Volatility patterns identification

#### Forecasting Results
- NBEATS model performance:
  - RMSE: 2.3%
  - MAE: 1.8%
  - R² Score: 0.89
- 7-day forecast accuracy: 92%
- 30-day forecast accuracy: 85%

### 4. News Impact Analysis

#### Correlation Findings
- Strong correlation (0.72) between negative news and price drops
- Lag effect of 2-3 days for major news impact
- Volume of coverage significantly affects price movement

#### Impact Metrics
- News sentiment as a leading indicator
- Trading volume increases 40% during high news periods
- Volatility correlation with news sentiment: 0.65

## Task Results and Visualization

### Individual Task Analysis
You can explore each task's detailed results through dedicated Jupyter notebooks:

1. **Task1_Analysis.ipynb**
   - Complete EDA findings
   - Text analysis visualizations
   - Sentiment distribution plots
   - Temporal pattern analysis

2. **Task2_Stock_Extraction.ipynb**
   - Stock ticker extraction results
   - Company name mapping accuracy
   - Mention frequency analysis
   - Validation metrics

3. **Task3_Financial_Analysis.ipynb**
   - Technical indicator results
   - Price trend visualizations
   - Volume analysis charts
   - Market statistics

4. **Task4-TimeSeries_Forecasting.ipynb**
   - NBEATS model training results
   - Prediction accuracy metrics
   - Model comparison charts
   - Performance visualizations

5. **Task5_News_Impact.ipynb**
   - News-price correlation analysis
   - Impact measurement results
   - Trading signal insights
   - Risk metric calculations

### Comprehensive Analysis
View the complete end-to-end analysis in `main.ipynb`, which integrates all tasks and provides:
- Combined results from all tasks
- Cross-task analysis
- Overall performance metrics
- Integrated visualizations

### Interactive Dashboard
Access our interactive Streamlit dashboard at http://localhost:8501 after running:
```bash
streamlit run app/streamlit_app.py
**https://resolved-merry-crappie.ngrok-free.app/**
```

The dashboard provides:
- Real-time data visualization
- Interactive model predictions
- Customizable analysis parameters
- Exportable reports and charts

## Project Structure
```
.
├── src/
│   ├── eda.py                  # EDA implementation
│   ├── stock_extraction.py     # Ticker extraction logic
│   ├── financial_analysis.py   # Financial analysis tools
│   ├── nbeats_trainer.py       # NBEATS model training
│   ├── nbeats_inference.py     # Model inference
│   ├── news_impact_analyzer.py # News impact analysis
│   └── sentiment_analysis.py   # Sentiment analysis
├── Task1_Analysis.ipynb
├── Task2_Stock_Extraction.ipynb                # Analysis notebooks
├── Task3_Financial_Analysis.ipynb
├── Task 4 - Time Series Forecasting.ipynb
├── Task5_News_Impact.ipynb
├── app/
│   └── streamlit_app.py       # Interactive dashboard
└── requirements.txt           # Project dependencies
```

## Execution Instructions

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Run Analysis Pipeline**
```bash
# Run EDA
python src/eda.py

# Extract stock tickers
python src/stock_extraction.py

# Perform financial analysis
python src/financial_analysis.py

# Train and run forecasting
python src/main.py
```

3. **View Results**
```bash
# Launch dashboard
streamlit run app/streamlit_app.py
Use this link to test the webapp: https://resolved-merry-crappie.ngrok-free.app/
```

## Key Dependencies

- pandas >= 2.0.0
- numpy >= 1.21.0
- scikit-learn >= 1.3.0
- tensorflow >= 2.13.0
- torch >= 2.1.2
- darts >= 0.24.0
- streamlit >= 1.15.0
- yfinance >= 0.2.31

## Results and Visualizations

The project generates comprehensive visualizations:
- Stock price trends and forecasts
- Sentiment distribution plots
- News impact correlation heatmaps
- Technical indicator charts
- Model performance metrics

All visualizations are available through the Streamlit dashboard.

## Conclusions and Recommendations

1. **Trading Insights**
   - Strong correlation between news sentiment and price movements
   - 2-3 day lag effect in news impact
   - Higher accuracy in short-term (7-day) forecasts

2. **Best Practices**
   - Monitor news sentiment as a leading indicator
   - Consider volume patterns in trading decisions
   - Use NBEATS forecasts for short to medium-term predictions

3. **Risk Management**
   - Higher volatility during negative news periods
   - Increased trading volume during major news events
   - Consider sentiment trends in position sizing

## Documentation

The code is extensively documented with:
- Detailed docstrings
- Function-level documentation
- Implementation notes
- Usage examples

For detailed API documentation, refer to individual module docstrings.

## Interactive Visualization Dashboard

Access the interactive dashboard at: http://localhost:8501 (after running Streamlit) **or** https://resolved-merry-crappie.ngrok-free.app/

### Dashboard Features
1. **News Analysis**
   - Real-time sentiment tracking
   - Word cloud visualization
   - Topic modeling results
   - Temporal patterns

2. **Stock Analysis**
   - Price charts and trends
   - Technical indicators
   - Volume analysis
   - Correlation plots

3. **Forecasting**
   - Interactive prediction plots
   - Confidence intervals
   - Model performance metrics
   - Backtesting results

4. **Impact Analysis**
   - News-price correlation
   - Trading signals
   - Risk indicators
   - Performance metrics

### Using the Dashboard
1. Navigate through different sections using the sidebar
2. Select stocks and date ranges
3. Adjust model parameters
4. Export visualizations and reports
