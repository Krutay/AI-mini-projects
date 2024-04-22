import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_samples = 50
X = np.linspace(0, 10, num_samples)
true_weights = np.array([2.5])
noise = 0.5 * np.random.randn(num_samples)
# ading some noise so the x to y relationship isn't perfectly linear
y = X * true_weights + 1 + noise
# true weight is w = [2.5] and bias term is 1 (effectively a 2-dim w = [1, 2.5])
# Add a column of ones to X for the bias term
X_bias = np.c_[np.ones(X.shape[0]), X]

true_weights = np.array([1,0.8])
true_weights_initial = true_weights
lamb = 0.01
learning_rate = 0.0001
iterations = 8000
for i in range(iterations):
    WX = np.dot(X_bias, true_weights)
    Y_WX = y - WX
    gradient = (-2) * np.dot(X_bias.T, Y_WX) + 2 * lamb * true_weights
    true_weights = true_weights - learning_rate * gradient

y_initial = np.dot(X_bias,true_weights_initial)
y_new = np.dot(X_bias,true_weights)
plt.scatter(X_bias[:,1], y, color="blue", label = "Training Points")
plt.plot(X,y_new,color = "red", label = "Ridge Regression")
plt.plot(X, y_initial,color="green",label = "Initial Line")
plt.title("Gradient Descent")
plt.legend()
plt.show()