import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.corpus import stopwords

# Download necessary NLTK data files if not already downloaded
nltk.download('stopwords')

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_text(text):
    """Clean the input text by removing stopwords and converting to lowercase."""
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_data(data):
    """Apply text preprocessing to the data."""
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    return data

def split_data(data):
    """Split the data into training and testing sets."""
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    return train_df, test_df

"""if __name__ == "__main__":
    # Load the raw data
    data = load_data('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/health_policies.csv')
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Split the data into training and testing sets
    train_df, test_df = split_data(processed_data)
    
    # Save the processed data
    train_df.to_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/train_data.csv', index=False)
    test_df.to_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/test_data.csv', index=False)"""
