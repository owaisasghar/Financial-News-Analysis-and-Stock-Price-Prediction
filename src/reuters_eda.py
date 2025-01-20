import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import spacy
from wordcloud import WordCloud

# Update NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Read the dataset
df = pd.read_csv('reuters_headlines.csv')

# Basic information about the dataset
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== First few rows ===")
print(df.head())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Convert Time to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%b %d %Y')

# Basic text analysis functions
def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def get_word_stats(text_series):
    # Combine all text
    all_text = ' '.join(text_series.apply(clean_text))
    
    # Just use word_tokenize directly
    words = word_tokenize(all_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return Counter(words)

# Analyze headlines
print("\n=== Headlines Analysis ===")
headline_word_freq = get_word_stats(df['Headlines'])
print("\nMost common words in headlines:")
print(pd.DataFrame(headline_word_freq.most_common(10), columns=['Word', 'Count']))

# Analyze descriptions
print("\n=== Descriptions Analysis ===")
desc_word_freq = get_word_stats(df['Description'])
print("\nMost common words in descriptions:")
print(pd.DataFrame(desc_word_freq.most_common(10), columns=['Word', 'Count']))

# Visualizations
plt.figure(figsize=(12, 6))
# Plot top 10 words in headlines
words, counts = zip(*headline_word_freq.most_common(10))
plt.bar(words, counts)
plt.title('Most Common Words in Headlines')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('headline_word_freq.png')
plt.close()

# Headlines per day
plt.figure(figsize=(10, 6))
df['Time'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Headlines per Day')
plt.xlabel('Date')
plt.ylabel('Number of Headlines')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('headlines_per_day.png')
plt.close()

# Headline length analysis
df['headline_length'] = df['Headlines'].str.len()
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='headline_length', bins=20)
plt.title('Distribution of Headline Lengths')
plt.xlabel('Number of Characters')
plt.tight_layout()
plt.savefig('headline_length_dist.png')
plt.close()

print("\n=== Headline Length Statistics ===")
print(df['headline_length'].describe())

# After loading the dataset, add these analyses:
print("\n=== Basic Dataset Statistics ===")
print(f"Total number of articles: {len(df)}")
print(f"Date range: from {df['Time'].min()} to {df['Time'].max()}")

# Analyze unique articles
print("\n=== Uniqueness Analysis ===")
print(f"Number of unique headlines: {df['Headlines'].nunique()}")
print(f"Number of unique descriptions: {df['Description'].nunique()}")

# Add sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Add sentiment scores
df['headline_sentiment'] = df['Headlines'].apply(get_sentiment)
df['description_sentiment'] = df['Description'].apply(get_sentiment)

print("\n=== Sentiment Analysis ===")
print("\nHeadline Sentiment Statistics:")
print(df['headline_sentiment'].describe())
print("\nDescription Sentiment Statistics:")
print(df['description_sentiment'].describe())

# Visualize sentiment distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['headline_sentiment'], bins=20)
plt.title('Headlines Sentiment Distribution')
plt.xlabel('Sentiment Score')

plt.subplot(1, 2, 2)
sns.histplot(df['description_sentiment'], bins=20)
plt.title('Descriptions Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.close()

# Analyze companies/organizations mentioned
nlp = spacy.load('en_core_web_sm')

def extract_organizations(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'ORG']

df['organizations'] = df['Headlines'].apply(extract_organizations)
all_orgs = [org for orgs in df['organizations'] for org in orgs]
org_freq = Counter(all_orgs)

print("\n=== Top Mentioned Organizations ===")
print(pd.DataFrame(org_freq.most_common(10), columns=['Organization', 'Count']))

# Add word cloud visualization
plt.figure(figsize=(12, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Headlines')
plt.tight_layout()
plt.savefig('headline_wordcloud.png')
plt.close() 