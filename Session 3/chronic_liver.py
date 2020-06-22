# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 23:28:50 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:49:11 2019

@author: Dell
#https://pymfe.readthedocs.io/en/latest/auto_examples/01_introductory_examples/plot_pymfe_default.html#sphx-glr-auto-examples-01-introductory-examples-plot-pymfe-default-py
"""
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
X1=pd.read_csv('C:/Users/Dell/Downloads/indian-liver-patient-records/indian_liver_patient.csv')
X=X1.iloc[:,3:10]
Y=X1.iloc[:,10]
feature=X.transpose()
feature=feature.describe()



X2=feature.transpose()
#X= np.concatenate((X,X2), 1)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
df1 = pd.DataFrame(X)
df2=pd.DataFrame(Y)
X=imp.fit_transform(df1)
Y=imp.fit_transform(df2)
#from pymfe.mfe import MFE
#mfe = MFE(groups=["statistical"])
#mfe.fit(X, Y)
#   
#ft_stat = mfe.extract()

#feature=X2.describe()

#df1 = pd.DataFrame(X)
#df2=pd.DataFrame(Y)
#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(strategy="most_frequent")
#X=imp.fit_transform(df1)
#Y=imp.fit_transform(df2)

#X=X.transpose()

#X = np.array(X)
#Y = np.array(Y)
#Features_statistics=[]
#from pymfe.mfe import MFE
#mfe = MFE(groups=["statistical"])
#for i in range(0,582):
#    
#    mfe.fit(X[i,:], Y[])
#    
#    ft_stat = mfe.extract()
#    Features_statistics.iloc[i:,]=ft_stat[1]
#    #print("\n".join("{:50} {:30}".format(X,Y) for X, Y in zip(ft_stat[0], ft_stat[1])))
#
#F_statistics=Features_statistics
#X=X.transpose()
#feature=X.describe()
#X=feature.transpose()
#import pandas as pd
#df1 = pd.DataFrame(X)
#df2=pd.DataFrame(Y)
#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(strategy="most_frequent")
#X=imp.fit_transform(df1)
#Y=imp.fit_transform(df2)
 
 
#from sklearn.preprocessing import Imputer 
#imputer= Imputer(missing_values='NaN', strategy='most_frequent' , axis = 1) 
#imputer.fit(X)
#X=imputer.transform(X)
#Y=imputer.transform(Y)


#imputer1= Imputer(missing_values='NaN', strategy='most_frequent' , axis = 1) 
#imputer1.fit(Y)
#d=imputer1.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
#classifier=svm.SVC()
##classifier=KNeighborsClassifier()
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#X_train=sc.fit_transform(X_train)
#X_test=sc.transform(X_test)
from sklearn.preprocessing import MinMaxScaler
sc_minmax=MinMaxScaler()
X_train=sc_minmax.fit_transform(X_train)
X_test=sc_minmax.transform(X_test)
#classifier=RandomForestClassifier(n_estimators=1000) 
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import accuracy_score
accuracies=cross_val_score(classifier,X=X,y=Y,cv=10,scoring='accuracy')
F1_score=cross_val_score(classifier,X=X,y=Y,cv=10,scoring='f1_macro')
from sklearn.metrics import accuracy_score
print("K fold accuracy")
print(accuracies.mean()*100,accuracies.std()*100)
print(confusion_matrix(y_test, y_pred))
#https://github.com/noobiecoder1942/Indian-Liver-Patient-Dataset/blob/master/ILPD.ipynb
#https://arxiv.org/ftp/arxiv/papers/1502/1502.05534.pdf