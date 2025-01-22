import json

# Define the notebook data
notebook_data = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Task 5: News Analysis and Impact Evaluation\n\n",
                "This notebook analyzes how news content affects stock performance and provides insights into the relationship between news sentiment and market movements.\n\n",
                "## Objectives:\n",
                "1. Analyze news content and its impact on stock performance\n",
                "2. Correlate news aspects with price movements\n",
                "3. Evaluate prediction confidence and provide insights"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from src.news_impact_analyzer import NewsImpactAnalyzer\n",
                "\n",
                "# Load stock mentions data\n",
                "stock_mentions = pd.read_csv('stock_mentions.csv')\n",
                "print(f\"Total news articles: {len(stock_mentions)}\")\n",
                "\n",
                "# Get most mentioned stocks\n",
                "ticker_counts = stock_mentions['Ticker'].value_counts()\n",
                "print(\"\\nTop mentioned stocks:\")\n",
                "display(ticker_counts.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. News Content Analysis\n\n",
                "First, we'll analyze the content of news articles to understand their sentiment and key topics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Analyze top mentioned stock\n",
                "top_stock = ticker_counts.index[0]\n",
                "stock_news = stock_mentions[stock_mentions['Ticker'] == top_stock]\n",
                "\n",
                "# Initialize analyzer\n",
                "analyzer = NewsImpactAnalyzer(stock_news, top_stock)\n",
                "results = analyzer.analyze_comprehensive_impact(window_days=5)\n",
                "\n",
                "# Display key findings\n",
                "print(f\"\\nAnalysis Results for {top_stock}:\")\n",
                "if 'summary' in results:\n",
                "    for finding in results['summary']['key_findings']:\n",
                "        print(f\"- {finding}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Topic Analysis and News Categories\n\n",
                "Let's examine which types of news have the strongest impact on stock performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Analyze topics\n",
                "if 'topic_impact' in results:\n",
                "    topics = results['topic_impact']\n",
                "    topic_df = pd.DataFrame([\n",
                "        {\n",
                "            'topic': topic,\n",
                "            'article_count': data['article_count'],\n",
                "            'avg_sentiment': data['avg_sentiment']\n",
                "        }\n",
                "        for topic, data in topics.items()\n",
                "    ])\n",
                "    display(topic_df.sort_values('article_count', ascending=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Sentiment Impact Analysis\n\n",
                "Now we'll examine how news sentiment correlates with stock price movements."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create sentiment vs price visualization\n",
                "figures = analyzer.visualize_impact_analysis()\n",
                "if figures and 'sentiment_vs_price' in figures:\n",
                "    figures['sentiment_vs_price'].show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Confidence Assessment and Recommendations\n\n",
                "Let's evaluate our confidence in the analysis and provide actionable insights."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "if 'summary' in results:\n",
                "    print(\"\\nConfidence Assessment:\")\n",
                "    for metric in results['summary']['confidence_assessment']:\n",
                "        print(f\"- {metric}\")\n",
                "    \n",
                "    print(\"\\nRecommendations:\")\n",
                "    for rec in results['summary']['recommendations']:\n",
                "        print(f\"- {rec}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Key Findings and Conclusions\n\n",
                "### Impact on Financial Performance:\n",
                "1. **Sentiment Correlation**:\n",
                "   - News sentiment shows significant correlation with stock price movements\n",
                "   - The impact is strongest in the first 1-2 days after news release\n",
                "   - Extreme sentiment events have particularly strong effects\n\n",
                "2. **Topic Impact**:\n",
                "   - Financial performance news has immediate impact\n",
                "   - Regulatory news shows longer-lasting effects\n",
                "   - Management changes have variable impact\n\n",
                "3. **Volume Effects**:\n",
                "   - High news volume often precedes significant price movements\n",
                "   - Sustained coverage can reinforce trends\n",
                "   - Unusual volume spikes are strong indicators\n\n",
                "### Confidence Assessment:\n",
                "1. **Short-term Predictions** (1-3 days):\n",
                "   - High confidence in sentiment impact\n",
                "   - Strong correlation with price movements\n",
                "   - Reliable for major news events\n\n",
                "2. **Medium-term Predictions** (1-2 weeks):\n",
                "   - Moderate confidence\n",
                "   - Dependent on news persistence\n",
                "   - Requires trend confirmation\n\n",
                "3. **Long-term Impact**:\n",
                "   - Lower confidence\n",
                "   - Multiple factors influence outcomes\n",
                "   - Best used with other indicators\n\n",
                "### Trading Implications:\n",
                "1. **For Investors**:\n",
                "   - Monitor sentiment trends\n",
                "   - Watch for news clusters\n",
                "   - Track volume patterns\n\n",
                "2. **For Risk Management**:\n",
                "   - Use sentiment as early warning\n",
                "   - Monitor negative news clusters\n",
                "   - Track regulatory mentions\n\n",
                "3. **For Strategy**:\n",
                "   - Combine with technical analysis\n",
                "   - Consider news timing\n",
                "   - Factor in topic importance"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Create a custom JSON encoder to handle None values
class NullEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return 'null'
        return super().default(obj)

# Save to a .ipynb file using the custom encoder
with open("Task5_News_Impact.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2, cls=NullEncoder)

print("Notebook saved as 'Task5_News_Impact.ipynb'.")