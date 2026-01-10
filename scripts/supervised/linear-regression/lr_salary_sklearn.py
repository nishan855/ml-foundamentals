from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from linear_model import LinearSalaryModel
import math


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

def plot_pytorch_training_hypothesis(X_train, y_train, model, title):
    plt.scatter(X_train, y_train, color="green", label="Training Data")
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    X_line_np = np.linspace(X_train_np.min(), X_train_np.max(), 100).reshape(-1, 1)
    X_line = torch.tensor(X_line_np, dtype=torch.float32)
    y_line = model(X_line).detach().numpy()
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


def run_linear_regression_pytorch():
    dataset = Path(__file__).parent / "../../../datasets/lr_salary_dataset.csv"
    df = pd.read_csv(dataset)
    df = df.dropna()
    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    linear_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    loss_history = []

    for epoch in range(100):
        y_pred = linear_model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())  

        if epoch < 5 or epoch % 20 == 0:
            w, b = linear_model.weight.item(), linear_model.bias.item()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")

    w, b = linear_model.weight.item(), linear_model.bias.item()
    print(f"Final weight: {w:.4f}")
    print(f"Final bias: {b:.4f}")

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    preds = linear_model(X_test_tensor)
    mse = mean_squared_error(y_test, preds.detach().numpy().flatten())
    rmse = math.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")
    print("Predictions:", preds.flatten().tolist())   

    plot_pytorch_training_hypothesis(
        X_train,
        y_train,
        linear_model,
        "Pytorch Linear Regression"
    )

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(loss_history)+1), loss_history, color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_linear_regression_pytorch()

    # run_closed_form()
    # run_sgd()
