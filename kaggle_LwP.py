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

def learning_with_prototype(X_train, y_train, prototype_points):
    # Calculate means
    mean_1 = X_train[y_train == 0].mean(axis=0)
    mean_2 = X_train[y_train == 1].mean(axis=0)

    # Calculate distances
    dist_1 = np.linalg.norm(prototype_points - mean_1, axis=1)
    dist_2 = np.linalg.norm(prototype_points - mean_2, axis=1)

    # Classify based on distances
    y_values = np.where(dist_1 > dist_2, 1, 0)

    return y_values

Z = learning_with_prototype(X_train, y_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Learning with Prototype (LwP) Classifier')
plt.legend()
plt.grid(True)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Class 0', marker='o', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Class 1', marker='+', edgecolors='k')
plt.show()