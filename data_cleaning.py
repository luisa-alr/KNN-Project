# Luisa Rosa
# HW 2 - Data Mining
# 03/04/2024

import csv
import numpy as np
import pandas as pd
from scipy.stats import zscore

# Data cleaning - open clean train and test data as a float np array
# train
train_file = open("spam_train.csv", "r")
train_list = list(csv.reader(train_file, delimiter=","))
train_file.close()
# remove first row with column names
train_list = train_list[1:]
# tranforming list into numpy array
train_array = np.array(train_list)
# turn elements in the array from string to float
train_array = train_array.astype(np.float)

# test
test_file = open("spam_test.csv", "r")
test_list = list(csv.reader(test_file, delimiter=","))
test_file.close()
# remove first row with column names
test_list = test_list[1:]
# tranforming list into numpy array
test_array = np.array(test_list)
# deleting ID column using np.delete()
test_array = np.delete(test_array, 0, axis=1)
# turn elements in the array from string to float
test_array = test_array.astype(np.float)

# Save train and test data as df
train_df = pd.read_csv("spam_train.csv")
test_df = pd.read_csv("spam_test.csv")

# Create a new DF for labels.
train_labels = train_df["class"]
test_labels = test_df["Label"]

# transform df in arrays
train_labels = np.array(train_labels).astype(float)
test_labels = np.array(test_labels).astype(float)

# Drop the ID columns from DF
test_df.drop(test_df.columns[0], axis=1, inplace=True)

# Z-score normalize test and train arrays
test_normalized = round(zscore(test_df), 2)
train_normalized = round(zscore(train_df), 2)

# get labels normalized values

test_labels_normalized = test_normalized.iloc[:, -1]

# transform df into np_array
test_normalized = np.array(test_normalized).astype(float)
train_normalized = np.array(train_normalized).astype(float)
test_labels_normalized = np.array(test_labels_normalized).astype(float)
