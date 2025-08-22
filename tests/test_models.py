import os
import pandas as pd
from src.model_training import train_models
from src.model_evaluation import evaluate_model

def test_model_training_and_evaluation(tmp_path):
    # Create dummy train data
    X_train = pd.DataFrame({"age": [29, 54, 45], "chol": [210, 250, 190]})
    y_train = pd.DataFrame({"target": [0, 1, 0]})

    X_train_path = tmp_path / "X_train.csv"
    y_train_path = tmp_path / "y_train.csv"
    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)

    model_dir = tmp_path / "models"
    os.makedirs(model_dir, exist_ok=True)

    # Train models
    train_models(str(X_train_path), str(y_train_path), str(model_dir))

    # Check models exist
    assert (model_dir / "logistic_regression.pkl").exists()
    assert (model_dir / "decision_tree.pkl").exists()
    assert (model_dir / "random_forest.pkl").exists()

    # Create dummy test data
    X_test = pd.DataFrame({"age": [40, 50], "chol": [200, 230]})
    y_test = pd.DataFrame({"target": [0, 1]})
    X_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Evaluate logistic regression model
    acc, report = evaluate_model(str(model_dir / "logistic_regression.pkl"),
                                 str(X_test_path), str(y_test_path))
    assert isinstance(acc, float)
    assert "precision" in report
