from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test