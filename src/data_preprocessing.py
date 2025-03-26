import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(url, data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    data_path = os.path.join(data_dir, "sms_spam.parquet")
    
    if not os.path.exists(data_path):
        try:
            data = pd.read_parquet(url)
            data.to_parquet(data_path)
            print(f"Dataset saved to {data_path}.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please download the dataset manually from this link and place it in the data/ folder:")
            print("https://huggingface.co/datasets/ucirvine/sms_spam")
            return None, None, None
    else:
        print(f"Dataset loaded from {data_path}.")
    
    # Load the dataset
    data = pd.read_parquet(data_path)
    
    # Print dataset information
    print("Dataset columns:", data.columns.tolist())
    print("Data types of each column:")
    print(data.dtypes)
    print("Unique values in each column:")
    for col in data.columns:
        print(f"{col}: {data[col].unique()[:5]}")
    
    # Identify label and message columns
    label_col = None
    message_col = None
    for col in data.columns:
        # Check for label column (can be string with 'ham'/'spam' or integer with 0/1)
        if pd.api.types.is_string_dtype(data[col]) and set(data[col].unique()).issubset({'ham', 'spam'}):
            label_col = col
        elif pd.api.types.is_integer_dtype(data[col]) and set(data[col].unique()).issubset({0, 1}):
            label_col = col
        # Check for message column (string type)
        elif pd.api.types.is_string_dtype(data[col]):
            message_col = col
    
    if label_col is None or message_col is None:
        print("Label or message columns not found.")
        return None, None, None
    
    # Rename columns
    data = data.rename(columns={label_col: 'label', message_col: 'message'})
    
    # If labels are 'ham'/'spam', map them to 0/1; otherwise, assume they are already 0/1
    if pd.api.types.is_string_dtype(data['label']):
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Preprocessing
    data['cleaned_message'] = data['message'].apply(preprocess_text)

    # Convert to Bag of Words model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['cleaned_message'])
    y = data['label']

    return X, y, vectorizer