from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from linear_model import LinearSalaryModel


def load_csv(filepath: str, features: list[str], target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath)
    return df[features], df[target]


def plot_training_hypothesis(X_train, y_train, model, title):
    plt.scatter(X_train, y_train, color="green", label="Training Data")

    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    plt.plot(X_line, y_line, color="red", linewidth=2, label="Regression Line")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_curve(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("SGD Training Loss per Epoch")
    plt.grid(True)
    plt.show()


def run_closed_form():
    dataset = Path(__file__).parent / "../../../datasets/lr_salary_dataset.csv"
    X, y = load_csv(dataset, ["YearsExperience"], "Salary")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSalaryModel()
    model.train_closed_form(X_train, y_train)

    preds = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    plot_training_hypothesis(
        X_train.values.flatten(),
        y_train,
        model,
        "Closed-Form Linear Regression"
    )


def run_sgd():
    dataset = Path(__file__).parent / "../../../datasets/lr_salary_dataset.csv"
    X, y = load_csv(dataset, ["YearsExperience"], "Salary")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSalaryModel()
    model.train_sgd(X_train, y_train, lr=0.01, epochs=40)

    preds = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    plot_training_hypothesis(
        X_train.values.flatten(),
        y_train,
        model,
        "SGD Regression"
    )
    plot_loss_curve(model.losses)


if __name__ == "__main__":
    run_closed_form()
    run_sgd()
