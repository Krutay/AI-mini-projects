import numpy as np
import matplotlib.pyplot as plt

num_samples = 100
# True means and covariance matrices of the Gaussians generating the data from two classes
class_1_mean = np.array([1.0, 1.0])
class_2_mean = np.array([-2.0, -2.0])
# Let's use non-spherical classes (non-identity covariance matrix for each Gaussian)
class_1_cov = np.array([[0.8, 0.4], [0.4, 0.8]])
class_2_cov = np.array([[0.8, -0.6], [-0.6, 0.8]])
X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, num_samples)
X_class_2 = np.random.multivariate_normal(class_2_mean, class_2_cov, num_samples)
X_train = np.vstack((X_class_1, X_class_2))
y_train = np.hstack((np.zeros(num_samples), np.ones(num_samples)))

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
np.arange(y_min, y_max, 0.02))

plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', marker='o', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', marker='+', edgecolors='k')

# Function which returns the class of each meshgrid point using KNN
def KNN(X_train, y_train, data_points):
    k = 6

    # Compute squared distances using broadcasting
    dist_sq = np.sum((X_train[:, None, :] - data_points[None, :, :]) ** 2, axis=2)

    # Find indices of k nearest neighbors using argpartition
    closest_indices = np.argpartition(dist_sq, kth=k, axis=0)[:k]

    # Use the indices to get the labels of the nearest neighbors
    y_values = y_train[closest_indices]

    # Count the number of 1s and 0s in each row
    counts = np.count_nonzero(y_values == 1, axis=0)

    # Assign the majority label to each prototype point
    y_values1 = np.where(counts > k / 2, 1, 0)

    return y_values1

Z = KNN(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Nearest Neighbour (KNN) Classifier')
plt.legend()
plt.grid(True)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', marker='o', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', marker='+', edgecolors='k')
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', marker='o', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', marker='+', edgecolors='k')
plt.show()