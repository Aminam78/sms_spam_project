# SMS Spam Detection Using Naive Bayes

## Overview
This is a simple project that is implemented for answering Problem 3 in the first assignment of "Text Mining" course.
This project implements an automated system to classify SMS messages into two categories: **Spam** and **Ham (non-spam)** using the **Naive Bayes** algorithm. The dataset used is the [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) from UCI, containing 5,574 labeled SMS messages.

## Project Structure


sms_spam_project/
│
├── data/
│   └── sms_spam.parquet      # Dataset file (automatically downloaded on first run)
│
├── src/
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── model_training.py     # Model training with Naive Bayes
│   ├── evaluation.py         # Model evaluation
│   └── predict.py            # Prediction on new messages
│
├── main.py                   # Main script to run the project
│
├── requirements.txt          # List of dependencies
│
└── README.md                 # Project documentation


## Installation
1. Clone the repository:
```bash
git clone https://github.com/Aminam78/sms_spam_project.git
cd sms_spam_project
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the project using the main script:
```bash
python main.py  # Run the project
```
This will:
- Download the dataset from Hugging Face using the URL: `hf://datasets/ucirvine/sms_spam/plain_text/train-00000-of-00001.parquet` on the first run and save it in the `data/` folder as a Parquet file.
- Load the dataset from the `data/` folder on subsequent runs.
- Preprocess the dataset.
- Train a Multinomial Naive Bayes model.
- Evaluate the model on the test set.
- Predict labels for new SMS messages.

## Example output
```bash
Dataset saved to data\sms_spam.parquet.
Dataset columns: ['sms', 'label']
Data types of each column:
sms      object
label     int64
dtype: object
Unique values in each column:
sms: ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n'
 'Ok lar... Joking wif u oni...\n'
 "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's\n"
 'U dun say so early hor... U c already then say...\n'
 "Nah I don't think he goes to usf, he lives around here though\n"]
label: [0 1]
Model Accuracy: 0.9721973094170404

Classification Report:
              precision    recall  f1-score   support

         Ham       0.99      0.98      0.98       954
        Spam       0.88      0.93      0.91       161

    accuracy                           0.97      1115
   macro avg       0.94      0.96      0.95      1115
weighted avg       0.97      0.97      0.97      1115

Message: Congratulations! You've won a $1000 gift card. Call now to claim!
Prediction: Spam

Message: Hey, are you free this evening for dinner?
Prediction: Ham

Message: Get a free iPhone now! Click here to claim your prize.
Prediction: Spam

Message: Reminder: Your appointment is tomorrow at 10 AM.
Prediction: Ham
```

## Methodology
1. **Data Preprocessing** (`data_preprocessing.py`):
   - Load the dataset from Hugging Face (on first run) or from the `data/` folder (on subsequent runs).
   - Automatically identify label and message columns, supporting both string labels ('ham', 'spam') and numeric labels (0, 1).
   - Preprocess the text: convert to lowercase, remove punctuation, tokenize, and remove stop words.
   - Convert messages to a Bag of Words model using `CountVectorizer`.

2. **Model Training** (`model_training.py`):
   - Split the data into training (80%) and test (20%) sets.
   - Train a Multinomial Naive Bayes model by calculating prior probabilities and conditional likelihoods for each word in the vocabulary.

3. **Evaluation** (`evaluation.py`):
   - Evaluate the model on the test set using metrics like Accuracy, Precision, Recall, and F1-Score.
   - Results: Accuracy ≈ 97.2%, Precision (Spam) ≈ 0.88, Recall (Spam) ≈ 0.93, F1-Score (Spam) ≈ 0.91.

4. **Prediction** (`predict.py`):
   - Test the model on new messages to classify them as Spam or Ham.

## Results
- The model achieved an accuracy of 97.2% on the test set.
- It correctly identifies spam messages with a precision of 0.88 and recall of 0.93
- The model successfully classified new messages, identifying promotional messages as Spam and conversational messages as Ham.

## Future Improvements
- Address class imbalance using techniques like SMOTE.
- Use TF-IDF instead of Bag of Words for better feature representation.
- Add features like message length or presence of URLs/numbers.
- Experiment with other models like SVM or neural networks.

## Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- pyarrow
- fsspec
- huggingface_hub

## Author
Amirhossein Amin Moghaddam
Master's Student in Computer Software Engineering at Iran University of Science and Technology (IUST)

Text Mining Course  
March 2025