import pandas as pd
from pathlib import Path
import torch
import torch.optim.sgd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


def load_csv_tensors():
    dataset = Path(__file__).parent / "../../../datasets/lr_student_performance.csv"
    df = pd.read_csv(dataset)

    X = df[
        [
            'Hours Studied',
            'Previous Scores',
            'Extracurricular Activities',
            'Sleep Hours',
            'Sample Question Papers Practiced'
        ]
    ].copy()

    y = df['Performance Index']

    X['Extracurricular Activities'] = (X['Extracurricular Activities'].map({
        'Yes': 1,
        'No': 0
    }).fillna(0))

    X = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor


def train_test_linear_grade_prediction_model():
    X,y = load_csv_tensors()
    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)
    model = torch.nn.Linear(5,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    criterion= torch.nn.MSELoss()
    loss_history =[]

    for epoch in range(200):
      y_pred = model(X_train)
      loss = criterion(y_pred,y_train)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_history.append(loss.item())
      if epoch < 5 or epoch % 20 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

    y_test_pred = model(X_test)
    y_test_pred_flatten = y_test_pred.detach().numpy().flatten()
    mse = mean_squared_error(y_test.flatten(),y_test_pred_flatten)
    print(f"RMSE on Test data : {math.sqrt(mse)}")
    

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(loss_history)+1), loss_history, color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()
    
    
if __name__ =="__main__":
    train_test_linear_grade_prediction_model()

    

