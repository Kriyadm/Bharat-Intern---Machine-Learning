import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import your KNN model and evaluation functions
from sklearn.neighbors import KNeighborsRegressor

# Load your dataset (assuming housing.csv is your dataset)
data = pd.read_csv('/content/housing.csv')

# Split data into features and target
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def knn_regression(k):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred

def evaluate_performance(pred, y_test):
    mae = np.mean(np.abs(pred - y_test))
    mse = np.mean((pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(pred - y_test) / y_test)
    return mae, mse, rmse, mape

def run_knn_and_evaluate(k):
    y_pred = knn_regression(k)
    mae, mse, rmse, mape = evaluate_performance(y_pred, y_test)
    return mae, mse, rmse, mape

def on_predict():
    k = int(k_entry.get())
    if k <= 0:
        messagebox.showerror("Error", "Please enter a valid number of neighbors (k > 0).")
        return
    
    mae, mse, rmse, mape = run_knn_and_evaluate(k)
    
    result_text.set(f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2%}")

# GUI setup
root = tk.Tk()
root.title("KNN Regression Evaluation")

mainframe = ttk.Frame(root, padding="20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

k_label = ttk.Label(mainframe, text="Enter the number of nearest neighbors (k):")
k_label.grid(column=0, row=0, sticky=tk.W)

k_entry = ttk.Entry(mainframe, width=10)
k_entry.grid(column=1, row=0)

predict_button = ttk.Button(mainframe, text="Predict", command=on_predict)
predict_button.grid(column=2, row=0)

result_text = tk.StringVar()
result_label = ttk.Label(mainframe, textvariable=result_text)
result_label.grid(column=0, row=1, columnspan=3, sticky=tk.W)

root.mainloop()
