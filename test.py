import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Salary_Dataset.csv")

# Features and target
X = df["YearsExperience"].values
y = df["Salary"].values

# Parameters
w = 0
b = 0

alpha = 0.001
epochs = 1000

n = len(X)

# Training
for epoch in range(epochs):

    dw = 0
    db = 0

    # Compute gradients using ALL samples
    for i in range(n):

        y_pred = w * X[i] + b
        error = y_pred - y[i]

        dw += error * X[i]
        db += error

    # Average gradients
    dw = (2 / n) * dw
    db = (2 / n) * db

    # Update parameters ONCE per epoch
    w = w - alpha * dw
    b = b - alpha * db

    # Print progress every 100 epochs
    if epoch % 100 == 0:

        cost = 0

        for i in range(n):
            pred = w * X[i] + b
            cost += (pred - y[i]) ** 2

        cost /= n

        print(f"Epoch {epoch}")
        print(f"w = {w:.4f}")
        print(f"b = {b:.4f}")
        print(f"Cost = {cost:.2f}")
        print()

print("Final Parameters")
print("w =", w)
print("b =", b)

# Predictions
predictions = w * X + b

# Sort for a clean regression line
idx = np.argsort(X)

# Plot
plt.scatter(X, y, label="Actual Data")
plt.plot(X[idx], predictions[idx], label="Regression Line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression using Batch Gradient Descent")
plt.legend()

plt.show()