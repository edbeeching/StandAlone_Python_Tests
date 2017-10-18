# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:25:03 2017

@author: Edward
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/Edward/Documents/GitHub/StandAlone_Python_Tests/jorge_census')
train = pd.read_csv('census_income_learn.csv', header=None)
test = pd.read_csv('census_income_test.csv', header=None)

train_desc = train.describe(include='all')
test_desc = train.describe(include='all')

train_dummy = pd.get_dummies(train.iloc[:,:-1])
train_y = train.iloc[:,-1]


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_dummy, train_y)

ros = RandomOverSampler(random_state=0)
Xr, Yr = ros.fit_sample(Xtrain, Ytrain)
Ybool = [0 if y == ' - 50000.' else 1 for y in Yr]

scl = StandardScaler()
Xrscl = scl.fit_transform(Xr)

clf = LogisticRegressionCV()
clf.fit(Xr, Ybool)

from sklearn.metrics import confusion_matrix
# Evaluation of Logistic regress on the oversampled data
preds = clf.predict(Xtest)
ytrue = [0 if y == ' - 50000.' else 1 for y in Ytest]

cm = confusion_matrix(ytrue, preds)















