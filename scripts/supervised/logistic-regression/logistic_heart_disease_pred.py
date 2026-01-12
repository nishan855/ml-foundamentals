from turtle import color
from typing import Any
import pandas as pd
from pathlib import Path
from pyparsing import col
from sklearn import linear_model
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import tensor
import torch
from torch import Tensor



def load_training_data()-> tuple[pd.DataFrame, pd.Series]:
    file_path = Path(__file__).parent / "../../../datasets/logistic_heart_disease.csv"
    df: pd.DataFrame = pd.read_csv(file_path)
    df.replace('NA',pd.NA,inplace= True)
    df = df.fillna(df.median())
    X: pd.DataFrame=df[["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]
    y: pd.Series= df["TenYearCHD"]
    return X,y



def logistic_regression_sklearn() -> None:
     X:pd.DataFrame
     y:pd.Series
     X,y= load_training_data()
     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     model = linear_model.LogisticRegression(solver="lbfgs",max_iter=5000)
     model.fit(X_train,y_train)
     preds = model.predict(X_test)
     print(f"Weights : {model.coef_}")
     print(f"ACC: {accuracy_score(y_test,preds)}")
     print(f"Confusion Matrix: {confusion_matrix(y_test,preds)}")
     print(f"Classification Report: {classification_report(y_test,preds)}")

def logistic_regression_pytorch() -> None:
    X, y = load_training_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss() 

    losses=[]

    for epoch in range(3000):
        optimizer.zero_grad()
        logits = torch.sigmoid(model(X_train))
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        logits = model(X_test)
        y_prob = torch.sigmoid(logits)
        y_pred = (y_prob >= 0.5).float()
        y_test_np = y_test.numpy()
        y_pred_np = y_pred.numpy()
        print(f"Accuracy: {accuracy_score(y_test_np, y_pred_np)}")
        print(f"Precision: {precision_score(y_test_np, y_pred_np)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test_np, y_pred_np)}")
        print(f"Classification Report:\n{classification_report(y_test_np, y_pred_np)}")
        plt.plot(range(1,len(losses)+1),losses,color="red")
        plt.show()

if __name__ =="__main__":
    logistic_regression_sklearn()
    logistic_regression_pytorch()    