## Data Mining - KNN Implementation from Scratch

## Luisa Rosa - Spring 2024

## Instructions:

- Download all files (4 Python programs and 2 CSV datasets)
- To see the answers to the questions, run the respective python program
  - Run 1a.py to get the test accuracies without normalization
  - Run 1b.py to get the test accuracies with z-score normalization
  - Run 1c.py to get the KNN predicted labels for the first 50 instances for each k value
- data_cleaning.py has all the data cleaning performed before the operations

---

## Question 1:

Implement the KNN classifier.

The implementation should accept two data files as input: 'spam_test.csv' and 'spam_train.csv'. Both files contain examples of e-mail messages, with each example having a class label of either “1” (spam) or “0” (no-spam).

Each example has 57(numeric) features that characterize the message. This program examines each example in the spam test set and classify it as one of the two classes, spam or no-spam. The classification is based on an unweighted vote of its k nearest examples in the spam train set.

All distances will be measured using regular Euclidean distance. Use of existing Python functions for normalzation and Euclidean distance calculations are allowed. However, the KNN implementation is done from scratch (without the use of KNN model Python libraries).

- a) Report test accuracies when k = 1, 5, 11, 21, 41, 61, 81, 101, 201, 401 without normalizing thefeatures.

- b) Report test accuracies when k = 1, 5, 11, 21, 41, 61, 81, 101, 201, 401 with z-score normal-ization applied to the features.

- c) In the (b) case, generate an output of KNN predicted labels for the first 50 instances (i.e.t1 - t50) when k = 1, 5, 11, 21, 41, 61, 81, 101, 201, 401 (in this order).

- d) What can you conclude by comparing the KNN performance in (a) and (b)?

### Solution:

1. Step 1 - Calculate the Euclidean Distance, a straight line distance between two vectors.
    - The smaller the value, the more similar two records will be.
    - 0 represents no difference.

2. Step 2 - Find K Nearest Neighbors
    - Calculate the distance between each record in the dataset to the new piece of data
    - Sort all of the records in the training dataset by their distance to the new data
    - Select the top k to return as the most similar neighbors

3. Step 3 - Make predictions
    - Extract the labels of all the k nearest neighbors found using the knn function
    - Make a prediction calculation the most common labels

4. Step 4 - Calculate the accuracy
    - Count each time the prediction matches the actual label
    - Divide the number of matches by the length of the test data

5. Step 5 - For each K value, find the test accuracies without normalization (a), with z-score normalization (b), or get the predicted labels for the first 50 instances (c).

---
