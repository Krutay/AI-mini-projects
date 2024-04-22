import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path to dataset 
file_path = 'TShirt_size.csv'

# Reading the CSV file
df = pd.read_csv(file_path)

X_train = np.array(df[['Height (in cms)', 'Weight (in kgs)']])
y_train = np.array(df['T Shirt Size'].map({'M': 0, 'L': 1}))

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
np.arange(y_min, y_max, 0.02))

plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', marker='o', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', marker='+', edgecolors='k')

def KNN(X_train, y_train, prototype_points):
    k = 6

    # Compute squared distances using broadcasting
    dist_sq = np.sum((X_train[:, None, :] - prototype_points[None, :, :]) ** 2, axis=2)

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
plt.show()