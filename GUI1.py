import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = load_iris()
x = data.data[:, 1:-1]  # Features (excluding first and last column)
y = data.target         # Labels

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

nclasses = np.unique(y_train).shape[0]
k = 3  # Default value for k

# Initialize arrays
dist = np.zeros(shape=x_train.shape[0])
pred = np.zeros(shape=x_test.shape[0])
classvotes = np.zeros(shape=nclasses)

# KNN prediction function
def knn_predict(test_data):
    global k, x_train, y_train
    dist = np.sqrt(np.sum((x_train - test_data)**2, axis=1))
    kminind = np.argpartition(dist, k)[:k]
    invdist = 1 / (dist + 1e-20)
    denom = np.sum(invdist[kminind])

    classvotes[:] = 0  # Reset classvotes
    for j in range(k):
        classvotes[int(y_train[kminind[j]])] += invdist[kminind[j]]

    classvotes /= denom
    return np.argmax(classvotes)

# Function to handle prediction
def handle_prediction():
    try:
        test_data = np.array([float(entry1.get()), float(entry2.get())]).reshape(1, -1)
        prediction = knn_predict(test_data)
        result_label.config(text=f'Predicted Class: {data.target_names[prediction]}')
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for the features")

# Creating the GUI
root = tk.Tk()
root.title("Iris Flower - KNN Classifier GUI")

tk.Label(root, text="Feature 1 (Sepal Width):").grid(row=0, column=0)
tk.Label(root, text="Feature 2 (Petal Length):").grid(row=1, column=0)

entry1 = tk.Entry(root)
entry2 = tk.Entry(root)

entry1.grid(row=0, column=1)
entry2.grid(row=1, column=1)

tk.Button(root, text="Predict", command=handle_prediction).grid(row=2, column=0, columnspan=2)

result_label = tk.Label(root, text="Predicted Class:")
result_label.grid(row=3, column=0, columnspan=2)

root.mainloop()
