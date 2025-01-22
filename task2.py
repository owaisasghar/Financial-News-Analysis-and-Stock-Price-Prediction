# Import necessary libraries
import pandas as pd
import spacy
import yfinance as yf
import requests
import os
from time import sleep
from typing import List, Optional
import re

# Load the spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = '1VEZ3497UNBK2A5Q'

def clean_company_name(name: str) -> str:
    """
    Clean and standardize company names.
    """
    # Remove common company suffixes
    suffixes = r'\s+(Corp\.?|Corporation|Inc\.?|Ltd\.?|Limited|LLC|Co\.?|Company|PLC|Group)\.?\s*$'
    name = re.sub(suffixes, '', name, flags=re.IGNORECASE)
    
    # Remove special characters and extra whitespace
    name = re.sub(r'[^\w\s-]', '', name)
    return ' '.join(name.split()).strip()

def extract_company_names(text: str) -> List[str]:
    """
    Use spaCy's Named Entity Recognition (NER) to extract company names from text.
    
    Args:
        text (str): Input text to extract company names from
        
    Returns:
        List[str]: List of extracted company names
    """
    if not isinstance(text, str):
        return []
    
    doc = nlp(text)
    companies = [clean_company_name(ent.text) for ent in doc.ents if ent.label_ == "ORG"]
    return list(set(company for company in companies if company))

def get_stock_ticker_alpha_vantage(company_name: str) -> Optional[str]:
    """
    Use Alpha Vantage's Symbol Search endpoint to map a company name to its stock ticker.
    
    Args:
        company_name (str): Name of the company
        
    Returns:
        Optional[str]: Stock ticker symbol if found, None otherwise
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": company_name,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "bestMatches" in data and data["bestMatches"]:
            return data["bestMatches"][0].get("1. symbol")
        return None
    except Exception as e:
        print(f"Alpha Vantage error for {company_name}: {e}")
        return None

def get_stock_ticker(company_name: str) -> Optional[str]:
    """
    Try multiple methods to get stock ticker for a company name.
    """
    # Try Alpha Vantage first
    ticker = get_stock_ticker_alpha_vantage(company_name)
    if ticker:
        return ticker
    
    # Try with cleaned name
    cleaned_name = clean_company_name(company_name)
    if cleaned_name != company_name:
        ticker = get_stock_ticker_alpha_vantage(cleaned_name)
        if ticker:
            return ticker
    
    # Try yfinance as last resort
    try:
        # Convert to potential ticker format
        ticker_guess = cleaned_name.upper().replace(' ', '')
        if len(ticker_guess) > 1:  # Only try if we have something meaningful
            ticker = yf.Ticker(ticker_guess)
            info = ticker.get_info()
            if info and 'symbol' in info:
                return info['symbol']
    except:
        pass
    
    return None

def process_headlines_data(input_file: str, output_file: str):
    """
    Process headlines data to extract companies and their stock tickers.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    try:
        # Read the CSV file
        print(f"Reading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Extract companies from headlines or description
        text_column = 'Description' if 'Description' in df.columns else 'Headlines'
        print(f"Extracting companies from {text_column}")
        df['Companies'] = df[text_column].apply(extract_company_names)
        
        # Create a dictionary to store company-to-ticker mappings
        company_to_ticker = {}
        
        # Process each unique company
        unique_companies = {company for companies in df['Companies'] for company in companies}
        total = len(unique_companies)
        print(f"Found {total} unique companies")
        
        for i, company in enumerate(unique_companies, 1):
            print(f"Processing {i}/{total}: {company}")
            if company not in company_to_ticker:
                ticker = get_stock_ticker(company)
                if ticker:
                    print(f"Found ticker: {ticker}")
                    company_to_ticker[company] = ticker
                sleep(0.5)  # Rate limit
        
        # Map companies to tickers in the DataFrame
        df['Tickers'] = df['Companies'].apply(
            lambda x: [company_to_ticker.get(company) for company in x if company_to_ticker.get(company)]
        )
        
        # Save the enhanced data
        print(f"Saving processed data to {output_file}")
        df.to_csv(output_file, index=False)
        print("Processing completed successfully")
        
        # Print summary
        success_rate = len(company_to_ticker) / total * 100
        print(f"\nProcessed {total} companies")
        print(f"Found tickers for {len(company_to_ticker)} companies")
        print(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    # Define input and output files
    input_file = "data/reuters_headlines.csv"
    output_file = "data/reuters_headlines_with_tickers.csv"
    
    # Process the data
    process_headlines_data(input_file, output_file)