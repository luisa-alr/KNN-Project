# Luisa Rosa
# HW 2 - Data Mining
# 03/04/2024

from math import sqrt
from collections import Counter
import numpy as np
from data_cleaning import train_normalized, test_normalized 

# Calculate the Euclidean distance between two vectors
def euclidean_dist(row1, row2):
    dist = 0.0
    for i in range(len(row1) - 1):
        dist += (row1[i] - row2[i]) ** 2
    return sqrt(dist)

# Locate the most similar k neighbors
def knn(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean_dist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_class(train, test_row, k):
    neighbors = knn(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    vote_counts = Counter(output_values)
    prediction = vote_counts.most_common(1)[0][0]
    return prediction

# Calculate Accuracy
def accuracy(predictions, test_class):
    correct = sum(
        1 for pred, true_label in zip(predictions, test_class) if pred == true_label
    )
    return correct / float(len(test_class)) * 100

# Test accuracies without normalization
k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

# QUESTION 1 C:
print("\nKNN predicted labels for the first 50 instances for each k value:")
i = 1
for test_row in test_normalized[:50]:
    prediction_values = list()
    for k in k_values:
        prediction = predict_class(train_normalized, test_row, k)
        if prediction == 1.24:
            prediction_values.append("spam")
        else:
            prediction_values.append("no")
    print(f"Row t{i}: {prediction_values}", sep = ", ")
    i += 1