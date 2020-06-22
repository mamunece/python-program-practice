# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 02:28:36 2020

@author: Dell
"""


s=[4, 5, 6, 7, 59, 64, 69, 74, 75, 84, 85, 98, 137, 147, 153, 161, 162, 163, 165, 178, 185, 197, 233, 238, 240]

m=[ 251,   319,   549 ,   31 ,  409 ,  237,   270,   548,   561,    21 ,  277,   405,   346,    9,   555,    86,   265,   502,    84,   343,547,  513,   249,   281,   415,   327,   494,    49,   255,   404,   560,   354,    71,   593,   283,   408,   498,    37,   553,   349, 257,   483,   432,   339,   546,   577,    33,   356,   510,   488]
f=[237,   561,    44,    86,   409,   398,   319,   669,   318,   474,    84,    60,   302,    85,    66,     6,   649,   528,   338,   293, 375,   371,   104,   698,    98,   200]
import timeit

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from enn.enn import ENN
 #label_binarize


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#conda install --user mlxtend
#from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np
from sklearn.model_selection import KFold
size = 10
cv = KFold(size, shuffle=True)
#from sklearn.model_selection import learning_curve


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from xgboost import XGBClassifier

#from sklearn import svm
import scipy.io
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import itertools    
from sklearn.model_selection import ShuffleSplit

#X=sc.fit_transform(X)
#from sklearn.model_selection import cross_val_score
#accuracies=cross_val_score(classifier,X=X,y=Y,cv=10)

data=pd.read_csv('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/Feature_User_all_tsne.csv')
from sklearn.utils import shuffle
#data=shuffle(data)
us1=data[data.iloc[:,703] == 1]
us2=data[data.iloc[:,703] == 2]
us3=data[data.iloc[:,703] == 3]
us4=data[data.iloc[:,703] == 4]
us5=data[data.iloc[:,703] == 5]
us6=data[data.iloc[:,703] == 6]
us7=data[data.iloc[:,703] == 7]
us8=data[data.iloc[:,703] == 8]
data.drop(data[data.iloc[:,703] ==1].index, inplace = True)
X=data.iloc[:,0:702]
X=X.iloc[:,m]
X=X.iloc[:,0:26]
Y=data.iloc[:,702]
Y=Y.astype(np.float64)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
#classifier=svm.SVC(probability=True,gamma='scale')
classifier=ENN()
#classifier=RandomForestClassifier()
#classifier=XGBClassifier()
#classifier=svm.SVC( C=12,gamma=0.0006, probability=True)
#classifier=svm.SVC( C= 1.8345947594465675,gamma=0.1, probability=True)


classifier.fit(X,Y)
X_test=us1.iloc[:,0:702]
X_test=X_test.iloc[:,m]                   #chan
X_test=X_test.iloc[:,0:26]
X_test=sc.fit_transform(X_test)
Y_test=us1.iloc[:,702]                  #change
y_pred=classifier.predict(X_test)
print("Accuracy/F1 score of the  model")

from sklearn.metrics import f1_score

print(accuracy_score(Y_test,y_pred)*100,'/',f1_score(Y_test,y_pred,average='macro')*100)
   

def plot_learning_curve(estimator, title, X, Y, ylim=None, cv=10,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
   
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.figure(dpi=100)
    return plt





title = "Learning Curves "
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.10 ,random_state=0)

estimator =classifier
#estimator=RandomForestClassifier()
plot_learning_curve(estimator, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)





