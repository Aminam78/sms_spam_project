import re
from nltk.tokenize import word_tokenize
from src.data_preprocessing import stop_words  # وارد کردن stop_words از data_preprocessing

def predict_new_messages(model, vectorizer, new_messages):
    cleaned_new_messages = [preprocess_text(msg) for msg in new_messages]
    X_new = vectorizer.transform(cleaned_new_messages)
    predictions = model.predict(X_new)

    for msg, pred in zip(new_messages, predictions):
        label = 'Spam' if pred == 1 else 'Ham'
        print(f"Message: {msg}\nPrediction: {label}\n")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)