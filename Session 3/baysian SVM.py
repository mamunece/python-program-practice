# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:20:32 2019

@author: Dell
"""
m=[ 251,   319,   549 ,   31 ,  409 ,  237,   270,   548,   561,    21 ,  277,   405,   346,    9,   555,    86,   265,   502,    84,   343,547,  513,   249,   281,   415,   327,   494,    49,   255,   404,   560,   354,    71,   593,   283,   408,   498,    37,   553,   349, 257,   483,   432,   339,   546,   577,    33,   356,   510,   488]
f=[237,   561,    44,    86,   409,   398,   319,   669,   318,   474,    84,    60,   302,    85,    66,     6,   649,   528,   338,   293, 375,   371,   104,   698,    98,   200]

import scipy.io
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
#mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/PAMAP_Features.mat')
#main_data=mat['PAMAP_Features']
#data=main_data[:,0:684]
#targets=main_data[:,684]
# mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/features_mrmr.mat')
# X=mat['selected_data']
# data=X[:,0:21]
# mat_2=scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/labels.mat')
# targets=mat_2['labels']
#mat_2=scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/labels_fcbf.mat')
#targets=mat_2['labels']
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#data=sc.fit_transform(data)
mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/Feature_User_all_tsne.mat')
data=mat['Feature_User_all_tsne']
targets=data[:,702]
data=data[:,m]
data=data[:,0:25]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data=sc.fit_transform(data)
x=data;
y=targets;


def svc_cv(C, gamma, data, targets):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, gamma=gamma, random_state=2,probability=True)
    cval = cross_val_score(estimator, data, targets,cv=5)
    return cval.mean()
def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='accuracy', cv=5)
    return cval.mean()
def optimize_svc(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=15)

    print("Final result:", optimizer.max)


def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)
    

    print("Final result:", optimizer.max)
if __name__ == "__main__":
    
    print(Colours.yellow("--- Optimizing SVM ---"))
    optimize_svc(data, targets)
#    print(Colours.green("--- Optimizing Random Forest ---"))
#    
#    optimize_rfc(data, targets)


