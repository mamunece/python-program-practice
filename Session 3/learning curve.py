# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:26:33 2019

@author: Dell
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np
size = 10
cv = KFold(size, shuffle=True)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from xgboost import XGBClassifier
#from sklearn import svm
import scipy.io
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
import itertools    
from sklearn.model_selection import ShuffleSplit
#X=sc.fit_transform(X)
from sklearn.model_selection import cross_val_score
#accuracies=cross_val_score(classifier,X=X,y=Y,cv=10)


mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/PAMAP_Features.mat')
data=mat['PAMAP_Features']
#dataset=pandas.read_csv('pamap_2.csv')
#X=dataset.iloc[:, 0:243]
X=data[:,0:684]
Y=data[:,684]
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.30,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(X)
#from sklearn.decomposition import KernelPCA
#kpca=KernelPCA(n_components=27,kernel='rbf')
#X=kpca.fit_transform(X)
def plot_learning_curve(estimator, title, X, Y, ylim=None, cv=10,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
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
    return plt





title = "Learning Curves "
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.10 ,random_state=0)

estimator =svm.SVC(probability=True,gamma='scale')
#estimator=RandomForestClassifier()
plot_learning_curve(estimator, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

#title = r"Learning Curves (XGBOOST- Baysian Optimization)"
## SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
#estimator = XGBClassifier(colsample_bytree=0.9845488975887966,gamma=0.006,max_depth=3,n_estimators=250,learning_rate=0.08,n_jobs=4,)# l rate 0.05 silo gamma=0.005933508022491596
#plot_learning_curve(estimator, title, X, Y, (0.5, 1.01), cv=cv, n_jobs=4)

plt.show()
#def plot_curve():
#    # instantiate
#    classifier =svm.SVC(probability=True,gamma='scale',C=2)
#
#    # fit
#    classifier.fit(X, Y)
#    train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, n_jobs=None, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    
#    plt.figure()
#    plt.title("SVM Classifier")
#    plt.legend(loc="best")
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    plt.gca().invert_yaxis()
#    
#    # box-like grid
#    plt.grid()
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    
#    # plot the average training and test score lines at each training set size
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#    
#    # sizes the window for readability and displays the plot
#    # shows error from 0 to 1.1
#    plt.Ylim(0,13)
#    plt.show()
##matplotlib inline
#plot_curve()