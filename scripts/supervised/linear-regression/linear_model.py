from sklearn.linear_model import LinearRegression
import pandas as pd 

class LinearSalaryModel:

    def __init__(self) -> None:
        self.model= LinearRegression()

    def train(self,X: pd.DataFrame, y: pd.Series) -> None: 
        self.model.fit(X,y)

    def predict(self, X: pd.DataFrame ):  
        return self.model.predict(X)


