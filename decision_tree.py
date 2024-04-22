import pandas as pd
import numpy as np
import math

# Defining structure for tree
class Node:
    def __init__(self, feature=None, class_label=None, subtrees=None):
        self.feature = feature
        self.class_label = class_label
        self.subtrees = subtrees or {}

# Creating function to calculate the entropy of a node
def calculate_entropy(node):
    if len(node) == 0:
        return 0

    temp = {}
    for i in list(set(node)):
        temp[i] = 0

    for j in node:
        temp[j] += 1

    entropy = 0
    for k in temp:
        probability = temp[k] / len(node)
        entropy -= probability * math.log2(probability)

    return entropy

# Creating function to calculate information gain of a node by subtracting the final entropy from the initial entropy
def calculate_information_gain(y, subsets_y):
    total_samples = len(y)

    if total_samples == 0:
        return 0

    # Calculate entropy before the split
    information_gain = calculate_entropy(y)

    # Calculate entropy after the split and subtract each sub-split
    for subset in subsets_y:
        subset_entropy = calculate_entropy(subset)
        information_gain = information_gain - (len(subset) / total_samples) * subset_entropy

    return information_gain

# Function to find the best split
def find_best_split(X, y, used_features):
    best_value = None
    max_information_gain = -float('inf')

    # Iterating for every feature in X
    for feature in X:
        if feature in used_features:
            continue

        unique_values = df[feature].unique()
        subsets = []
        for value in unique_values:
            subset = df.loc[df[feature] == value, 'Drug'].tolist()
            subsets.append(subset)

        # Calculating information gains for all of them
        information_gain = calculate_information_gain(y, subsets)
        if information_gain > max_information_gain:
            # Maximizing
            max_information_gain = information_gain
            best_value = feature

    # Returning the best value
    return best_value

# Implementing function to create decision tree
def decision_tree(X, y, node, used_features=None):
    # Creating feature set to ensure features are not reused in subsequent depths
    if used_features is None:
        used_features = set()

    # Base case for recursion if entropy is close to 0 (due to python approximations)
    if 0 <= calculate_entropy(y) <= 0.001:
        node.class_label = y.mode().iloc[0]
        return

    # Finding the best split at a particular node
    best_feature = find_best_split(X, y, used_features)

    # If there can be no further splits 
    if best_feature is None:
        node.class_label = y.mode().iloc[0]
        return  

    # Adding the feature to the set of used features
    used_features.add(best_feature)
    node.feature = best_feature

    # Performing the same recursive operations for the subtrees
    unique_values = df[best_feature].unique()
    for value in unique_values:
        # Dropping the feature which is already used
        subset_indices = X[X[best_feature] == value].index
        subset_X = X.loc[subset_indices].drop(columns=best_feature)
        subset_y = y.loc[subset_indices]

        node.subtrees[value] = Node()
        decision_tree(subset_X, subset_y, node.subtrees[value], used_features.copy())

# Printing the structure of the decision tree 
def print_tree(node, depth=0):
    indent = "  " * depth
    if node.feature is not None:
        print(f"{indent}Feature: {node.feature}")
    else:
        print(f"{indent}Class: {node.class_label}")

    for value, subtree in node.subtrees.items():
        print(f"{indent}  Value: {value}")
        print_tree(subtree, depth + 1)

# Creating function to predict a sample by traversing the decision tree
def predict_sample(sample, node):
    # If it is a leaf node then return the label of the class
    if node.feature is None:  
        return node.class_label

    # Recursively entering the subtree for every subsequent feature
    feature_value = sample[node.feature]
    if feature_value in node.subtrees:
        subtree = node.subtrees[feature_value]
        return predict_sample(sample, subtree)
    else:
        # Returning default prediction if feature value is not found
        return 'DefaultPrediction'

# Path to CSV file
file_path = '/Users/siddharthsingh/Downloads/Assignment5_AI/drug200.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Preprocess the data
df['Age'] = pd.cut(df['Age'], bins=[0, 30, 50, float('inf')], labels=['LOW', 'MEDIUM', 'HIGH'])

# Store the data in X and y
X = df[['Age', 'Sex', 'BP', 'Cholesterol']]
y = df['Drug']

# Splitting the data into training and testing (80-20 split)
X_train = X.iloc[0:160, :]
y_train = y.iloc[0:160]

# Printing the tree structure
tree = Node()
decision_tree(X_train, y_train, tree)

# Print the tree
print("Tree structure:")
print_tree(tree)

# Testing accuracy
correct_predictions = 0
total_predictions = 0

# Testing for rows 160 to 200 of the dataset
for sample_row_index in range(160, 200):
    total_predictions += 1
    sample_features = X.iloc[sample_row_index].to_dict()
    prediction = predict_sample(sample_features, tree)
    if prediction == df.loc[sample_row_index, 'Drug']:
        correct_predictions += 1

# Printing accuracy
accuracy = (correct_predictions/total_predictions)*100
print("Accuracy: " + str(accuracy) + "%")