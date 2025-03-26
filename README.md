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
└── README.md


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

## Methodology
1. **Data Preprocessing** (`data_preprocessing.py`):
   - Load the dataset from Hugging Face (on first run) or from the `data/` folder (on subsequent runs), and map labels (`ham` → 0, `spam` → 1).
   - Preprocess the text: convert to lowercase, remove punctuation, tokenize, and remove stop words.
   - Convert messages to a Bag of Words model using `CountVectorizer`.

2. **Model Training** (`model_training.py`):
   - Split the data into training (80%) and test (20%) sets.
   - Train a Multinomial Naive Bayes model by calculating prior probabilities and conditional likelihoods for each word in the vocabulary.

3. **Evaluation** (`evaluation.py`):
   - Evaluate the model on the test set using metrics like Accuracy, Precision, Recall, and F1-Score.
   - Results: Accuracy ≈ 98%, Precision (Spam) ≈ 0.93, Recall (Spam) ≈ 0.92, F1-Score (Spam) ≈ 0.92.

4. **Prediction** (`predict.py`):
   - Test the model on new messages to classify them as Spam or Ham.

## Results
- The model achieved an accuracy of 98% on the test set.
- It correctly identifies spam messages with a precision of 0.93 and recall of 0.92.
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