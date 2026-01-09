import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


class LinearSalaryModel:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.losses = []

    def train_closed_form(self, X: pd.DataFrame, y: pd.Series):
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.model_type = "cf"

    def train_sgd(self, X: pd.DataFrame, y: pd.Series, lr=0.01, epochs=20):
        self.model = SGDRegressor(
            learning_rate="constant",
            eta0=lr,
            max_iter=1,
            tol=None,
            random_state=42
        )

        self.losses = []

        for _ in range(epochs):
            self.model.partial_fit(X, y)
            preds = self.model.predict(X)
            loss = mean_squared_error(y, preds)
            self.losses.append(loss)

        self.model_type = "sgd"

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        return self.model.predict(X)
