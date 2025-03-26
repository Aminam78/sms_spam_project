from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.predict import predict_new_messages

# Dataset URL
url = "hf://datasets/ucirvine/sms_spam/plain_text/train-00000-of-00001.parquet"

# Path to the data directory
data_dir = "data"

# Load and preprocess the data
X, y, vectorizer = load_and_preprocess_data(url, data_dir)

# Check if dataset loading failed
if X is None:
    exit()

# Train the model
model, X_train, X_test, y_train, y_test = train_model(X, y)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Predict on new messages
new_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim!",
    "Hey, are you free this evening for dinner?",
    "Get a free iPhone now! Click here to claim your prize.",
    "Reminder: Your appointment is tomorrow at 10 AM."
]
predict_new_messages(model, vectorizer, new_messages)