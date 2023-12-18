# -*- coding: utf-8 -*-
"""
@author: lyu
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils_1_read_excel_table import read_table

Data_Logit = read_table(r'C:\Users\lyu\Desktop\Spreadsheets\Quant_Logit_and_Probit.xlsm', "Table1")
print(Data_Logit)

# Define the dependent variable and the independent variables
X = Data_Logit[Data_Logit.columns[3:]]
y = Data_Logit['Default']

# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Perform the logistic regression
model = sm.Logit(y, X)
result = model.fit()

# Print the summary of the regression
print(result.summary())

# Get the predicted probabilities
y_hat = result.predict(X)

# Print the predicted probabilities
print(y_hat)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Convert the predicted probabilities to binary predictions
# by thresholding at 0.5
y_pred = (y_hat > 0.5).astype(int)

# Calculate the performance metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auroc = roc_auc_score(y, y_hat)

print(accuracy)
print(precision)
print(recall)
print(f1)
print(auroc)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
