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

X_transpose =  np.transpose(X_bias)
XX_transpose = np.dot(X_transpose,X_bias)
lambda_identity = 0.01 * np.identity(2)
XX_transpose_identity = np.add(XX_transpose ,lambda_identity)
XX_transpose_inverse = np.linalg.inv(XX_transpose_identity)
XX_transpose_inverse_X_transpose = np.dot(XX_transpose_inverse,X_transpose)
w = np.dot(XX_transpose_inverse_X_transpose,y)

y_plot = w[0]  + np.dot(w[1],X)

plt.scatter(X_bias[:,1], y, label='X_bias vs y')
plt.plot(X, y_plot, label = 'X vs y_plot')

plt.legend()
plt.show()