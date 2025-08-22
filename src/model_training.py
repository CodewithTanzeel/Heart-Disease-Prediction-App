import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train_path, y_train_path, model_dir):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    models = {
        "logistic_regression.pkl": LogisticRegression(max_iter=1000),
        "decision_tree.pkl": DecisionTreeClassifier(),
        "random_forest.pkl": RandomForestClassifier()
    }

    for filename, model in models.items():
        model.fit(X_train, y_train)
        with open(f"{model_dir}/{filename}", "wb") as f:
            pickle.dump(model, f)
