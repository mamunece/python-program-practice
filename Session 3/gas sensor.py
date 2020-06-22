# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:28:20 2019

@author: Dell
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
dataset=pd.read_csv('PS1_with label.csv')
X=dataset.iloc[:, 0:6000]
Y=dataset.iloc[:,6000]
X=X.transpose()
feature=X.describe()
X=feature.transpose()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.30,random_state=0,stratify=Y)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
classifier=svm.SVC(probability=True,gamma='scale')
#classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
print(accuracy_score(Y_test,y_pred)*100)



