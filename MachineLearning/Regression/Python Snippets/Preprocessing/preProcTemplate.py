# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 13:25:27 2018

@author: Harry
"""

#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0, verbose=1)
#You need to fit first and then apply the object to whatever corps of data you want
X[:, 1:3] = imputer.fit_transform(X[:,1:3])

#Encoding categorical vars
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])
#LabelEncoder => Transforms strings to integers
#OneHotEncoder => Transforms integers to dummy vars
#Remember to tranform it to array or X will become something else (something sinister)!
X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray()
y = LabelEncoder().fit_transform(y)

#Split to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
#Scale all except dummies
#If u have initiated the object u only need to fit once
#Otherwise you have to fit_transform for both sets
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

