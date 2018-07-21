#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:34:33 2018

@author: raksha
"""

# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

#dropping columns which are not required
dataset = dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1)
dataset_test = dataset_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Taking care of missing data
dataset.fillna(method='ffill', inplace=True)
dataset_test.fillna(method='ffill', inplace=True)

"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 3:4])
X[:, 3:4] = imputer.transform(X[:, 3:4])
imputer = imputer.fit(X[:, 0])
y[:, 0] = imputer.transform(X[:, 0])"""


# Encoding categorical data
# Encoding the Independent Variable

dataset=dataset.replace(['male','female'],[0,1])
dataset_test=dataset_test.replace(['male','female'],[0,1])

dataset=dataset.replace(['S','C','Q'],[0,1,2])
dataset_test=dataset_test.replace(['S','C','Q'],[0,1,2])

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
#regressor.fit(X, y)

# Predicting a new result
classifier = classifier.fit(dataset.ix[:,'Pclass':], dataset['Survived'])
randomforest = classifier.predict(dataset_test.ix[:,'Pclass':])
randomforest_output = pd.DataFrame(dataset_test['PassengerId'].values, columns=['PassengerId'])
randomforest_output['Survived'] = randomforest


"""y_pred = regressor.predict(6.5)"""


randomforest_output.to_csv('randomforestsubmission.csv', index=False)

