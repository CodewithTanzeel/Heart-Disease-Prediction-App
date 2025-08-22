import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, X_test_path, y_test_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return acc, report
