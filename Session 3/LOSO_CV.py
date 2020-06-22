# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 05:28:02 2020

@author: Dell
"""

gkf = GroupKFold(n_splits=8)

for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train, X_test, y_train, y_test)
    classifier=svm.SVC(probability=True,gamma='scale')
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    print(accuracy_score(y_test,y_pred)*100)