import pandas as pd
import numpy as np

# Defining sigmoid function
def sigmoid(z):
    # Making sure z is a numpy array
    z = np.array(z, dtype=np.float64)  
    return 1 / (1 + np.exp(-z))

# Performing gradient descent
def gradient_descent(X, y, weight, alpha, num_iterations):
    m = len(y)

    for iteration in range(num_iterations):
        predictions = X.dot(weight)
        errors = predictions - y
        gradient = X.T.dot(errors) / m
        weight = np.subtract(weight, alpha * gradient)

    # Returning the weight vector
    return weight

# Load the dataset
file_path = '/Users/siddharthsingh/Downloads/Assignment5_AI/drug200.csv'
df = pd.read_csv(file_path)

# Preprocess the data
df['Age'] = pd.cut(df['Age'], bins=[0, 30, 50, float('inf')], labels=['LOW', 'MEDIUM', 'HIGH'])

# Convert drug values to numerical labels
drug_mapping = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4}
df['Drug'] = df['Drug'].map(drug_mapping)

# Extract features and labels
X = df[['Age', 'Sex', 'BP', 'Cholesterol']]
y = df['Drug']

# Splitting the data into training and testing (80-20 split)
X_train = X.iloc[0:160, :]
y_train = y.iloc[0:160]

# Converting the values of the dataset into a feature set with true and false values if the features are present or not present respectively
X_one_hot = pd.get_dummies(X_train)

# Convert to numpy arrays
X_array = X_one_hot.values
y_array = pd.get_dummies(y_train).values

# Initialize parameters
num_features = X_array.shape[1]
num_classes = len(df['Drug'].unique())

# Initializing weight vector to 0
weight = np.zeros((num_features, num_classes))

# Set hyperparameters
alpha = 0.3
num_iterations = 2000

# Perform gradient descent
weight_vector = gradient_descent(X_array, y_array, weight, alpha, num_iterations)

# Calculating the matrix multiplication of the input array and the weight matrix
results = np.dot(X_array, weight_vector)

# Applying sigmoid function to get the probability of each class (logistic regression)
result = np.apply_along_axis(sigmoid, axis=1, arr=results)

# Taking last 40 samples as test data
X_test = X.iloc[160:200, :]
y_test = y.iloc[160:200]

# Creating a list of predictions 
predictions = []
X_one_hot_test = pd.get_dummies(X_test)

# Creating test arrays
X_array_test = X_one_hot_test.values
results_test = np.dot(X_array_test, weight_vector)

# Using argmax function to get the corresponding drug by taking the index of the class with maximum probability (Softmax)
for result in results_test:
    predictions.append(np.argmax(result))

# Test data for y (last 40 samples)
test = y.tolist()[160:200]

# Calculating accuracy
correct_predictions = 0
total_predictions = 0
for index in range(0, 40):
    total_predictions += 1
    if (predictions[index] == test[index]):
        correct_predictions += 1

# Printing the accuracy
accuracy = correct_predictions/total_predictions
print(accuracy)


'''Justification: Logistic regression can be used to predict the correct drug for a given sample 
but the accuracy is slighly low because the number of sample space is small.'''

    