# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:23:47 2024

@author: argad
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("C:/DecisionTree/credit.csv")

#data prepartion
#ckeck for null values

data.isnull().sum()
data.dropna()
data.columns

#there are 9 columns having non-numeric values,let's us now 
#there is ione column called which is not useful

data=data.drop(["phone"],axis=1)

#now there are 16 columns
lb=LabelEncoder()

data["checking_balance"]=lb.fit_transform(data["checking_balance"])
data["credit_history"]=lb.fit_transform(data["credit_history"])
data["purpose"]=lb.fit_transform(data["purpose"])
data["savings_balance"]=lb.fit_transform(data["savings_balance"])
data["employment_duration"]=lb.fit_transform(data["employment_duration"])
data["other_credit"]=lb.fit_transform(data["other_credit"])
data["housing"]=lb.fit_transform(data["housing"])
data["job"]=lb.fit_transform(data["job"])

#check for numeric columns

non_numeric_cols=data.select_dtypes(include=['object']).columns
print(non_numeric_cols)

data["checking_balance"]=lb.fit_transform(data["checking_balance"])
data["default"]=lb.fit_transform(data["default"])


#now let's check the how many unique values are there in target col
data["default"].unique()
data["default"].value_counts()

##now we want to split tree we need all feature columns
colnames=list(data.columns)

#now let's assign these columns to the variable predictor

predictors=colnames[:15]
target=colnames[15]

#spliting data into training and testing data set

from sklearn.model_selection import train_test_split

train,test=train_test_split(data,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)

model=DT(criterion='entropy')
model.fit(train[predictors],train[target])


#predictions on test data

preds=model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=["Actual"],colnames=["Predictions"])

np.mean(preds==test[target])

preds=model.predict(train[predictors])

pd.crosstab(train[target],preds,rownames=['Actual'],colnames=['Predictions'])

np.mean(preds == train[target])

