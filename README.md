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

5. Step 5 - For each K value, find the test accuracies without normalization (a), with z-score normalization (b), or get the predicted labels for the first 50 instances

---

### Answers:

a) Test accuracies without normalization:
    For k = 1, Accuracy: 75.22816166883963%
    For k = 5, Accuracy: 75.48891786179922%
    For k = 11, Accuracy: 76.48848326814428%
    For k = 21, Accuracy: 74.6631899174272%
    For k = 41, Accuracy: 75.22816166883963%
    For k = 61, Accuracy: 73.75054324206867%
    For k = 81, Accuracy: 72.66405910473706%
    For k = 101, Accuracy: 72.88135593220339%
    For k = 201, Accuracy: 73.14211212516297%
    For k = 401, Accuracy: 71.96870925684486%

b) Test accuracies with z-score normalization:
    For k = 1, Accuracy: 82.3554976097349%
    For k = 5, Accuracy: 83.61581920903954%
    For k = 11, Accuracy: 87.44024337244676%
    For k = 21, Accuracy: 87.1360278139939%
    For k = 41, Accuracy: 87.09256844850066%
    For k = 61, Accuracy: 87.1360278139939%
    For k = 81, Accuracy: 86.87527162103433%
    For k = 101, Accuracy: 86.39721860060843%
    For k = 201, Accuracy: 84.65884398087788%
    For k = 401, Accuracy: 81.48631029986963%

c) KNN predicted labels for the first 50 instances for each k value:
    Row t1: ['spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no', 'no']
    Row t2: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no', 'no']
    Row t3: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t4: ['spam', 'spam', 'spam', 'spam', 'no', 'no', 'spam', 'spam', 'spam', 'spam']
    Row t5: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t6: ['spam', 'spam', 'spam', 'no', 'no', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t7: ['spam', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t8: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t9: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t10: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t11: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t12: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t13: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no']
    Row t14: ['no', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t15: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t16: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t17: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t18: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no', 'no']
    Row t19: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t20: ['no', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t21: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t22: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no']
    Row t23: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t24: ['no', 'no', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t25: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t26: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t27: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t28: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t29: ['spam', 'spam', 'spam', 'no', 'spam', 'spam', 'spam', 'spam', 'no', 'no']
    Row t30: ['spam', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t31: ['spam', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t32: ['spam', 'spam', 'spam', 'spam', 'no', 'spam', 'spam', 'spam', 'no', 'no']
    Row t33: ['spam', 'spam', 'spam', 'spam', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t34: ['spam', 'spam', 'no', 'spam', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t35: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t36: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t37: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t38: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t39: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t40: ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t41: ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t42: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'no', 'no']
    Row t43: ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t44: ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no']
    Row t45: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t46: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t47: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t48: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t49: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']
    Row t50: ['spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam']

d) Using z-score normalization makes KNN work better, giving higher accuracies across different k values.


