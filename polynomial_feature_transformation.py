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

num_samples = 80

X = np.linspace(0, 5, num_samples)
X_bias = np.c_[np.ones(X.shape[0]), X]
X_sin = np.sin(X)

noise = np.random.normal(0, 0.5, num_samples)

Y = X_sin + noise
X_term = X_bias
degree = 3
for i in range(2,degree+1):
    X_term = np.c_[X_term, X**i]

lambd = 0.01
XTX = np.dot(X_term.T,X_term)
XTX_I = XTX + lambd * np.identity(XTX.shape[0])

XTX_I_INV = np.linalg.inv(XTX_I)

XTX_I_INV_XT = np.dot(XTX_I_INV,X_term.T)

Weight = np.dot(XTX_I_INV_XT,Y)

print(X_term.shape)
Y_plot = np.dot(X_term,Weight)
plt.plot(X,Y_plot,color = "red",label = "Ridge Regression")
plt.scatter(X, Y, label = "Training Points")
plt.plot(X,X_sin, label = "True Result",color = "green")
plt.title("Ridge Regression for non linear points")
plt.legend()
plt.show()