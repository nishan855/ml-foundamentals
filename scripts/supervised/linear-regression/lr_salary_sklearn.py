import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from linear_model import LinearSalaryModel  # your model class


def load_csv(filepath: str, features: list[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath)
    X = df[features]
    y = df[target]
    return X, y


def plot_training_hypothesis(X_train_values, y_train, model):
    """
    Plot training data and the learned hypothesis (regression line) from training data.
    """
    plt.scatter(X_train_values, y_train, color='green', label='Training Data')
    X_line = np.linspace(X_train_values.min(), X_train_values.max(), 100).reshape(-1, 1)
    y_line = model.predict(pd.DataFrame(X_line, columns=['YearsExperience']))
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Learned Hypothesis')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Linear Regression: Training Data and Hypothesis')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "../../../datasets/lr_salary_dataset.csv"
    X, y = load_csv(dataset_path, ["YearsExperience"], "Salary")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearSalaryModel()
    print("Training the model >>>")
    model.train(X_train, y_train)
    X_train_values = X_train.values.flatten()
    print("Predicting the test set >>>")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    X_test_values = X_test.values.flatten()
    print("\nFirst 5 inputs and predicted salaries:")
    for inp, pred in zip(X_test_values[:5], predictions[:5]):
        print(f"YearsExperience: {inp}, Predicted Salary: {pred:.2f}")

    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"R2 score: {r2:.2f}")
    plot_training_hypothesis(X_train_values, y_train, model)



if __name__ == "__main__":
    main()
