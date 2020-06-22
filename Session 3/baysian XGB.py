# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:20:32 2019

@author: Dell
"""
import scipy.io
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import StratifiedShuffleSplit

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/PAMAP_Features.mat')
main_data=mat['PAMAP_Features']
data=main_data[:,0:684]
targets=main_data[:,684]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data=sc.fit_transform(data)
#def svc_cv(C, gamma, data, targets):
#    """SVC cross validation.
#    This function will instantiate a SVC classifier with parameters C and
#    gamma. Combined with data and targets this will in turn be used to perform
#    cross validation. The result of cross validation is returned.
#    Our goal is to find combinations of C and gamma that maximizes the roc_auc
#    metric.
#    """
#    estimator = SVC(C=C, gamma=gamma, random_state=2,probability=True)
#    cval = cross_val_score(estimator, data, targets,cv=5)
#    return cval.mean()
#def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
#    """Random Forest cross validation.
#    This function will instantiate a random forest classifier with parameters
#    n_estimators, min_samples_split, and max_features. Combined with data and
#    targets this will in turn be used to perform cross validation. The result
#    of cross validation is returned.
#    Our goal is to find combinations of n_estimators, min_samples_split, and
#    max_features that minimzes the log loss.
#    """
#    estimator = RFC(
#        n_estimators=n_estimators,
#        min_samples_split=min_samples_split,
#        max_features=max_features,
#        random_state=2
#    )
#    cval = cross_val_score(estimator, data, targets,
#                           scoring='neg_log_loss', cv=5)
#    return cval.mean()
def xgb_cv(max_depth, gamma, colsample_bytree,data, targets):
    estimator = xgb(
            
            
        
        n_estimators=250,learning_rate=0.08,n_jobs=4,
        max_depth=max_depth,
        gamma=gamma,
        colsample_bytree=colsample_bytree,
       
    )
    
##    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#    scores=cross_val_score(classifier, X, Y, cv=cv)
    cval = cross_val_score(estimator, data, targets, scoring='neg_log_loss',
                            cv=5)
    return cval.mean()
    
#def optimize_svc(data, targets):
#    """Apply Bayesian Optimization to SVC parameters."""
#    def svc_crossval(expC, expGamma):
#        """Wrapper of SVC cross validation.
#        Notice how we transform between regular and log scale. While this
#        is not technically necessary, it greatly improves the performance
#        of the optimizer.
#        """
#        C = 10 ** expC
#        gamma = 10 ** expGamma
#        return svc_cv(C=C, gamma=gamma, data=data, targets=targets)
#
#    optimizer = BayesianOptimization(
#        f=svc_crossval,
#        pbounds={"expC": (-3, 2), "expGamma": (-4, -1)},
#        random_state=1234,
#        verbose=2
#    )
#    optimizer.maximize(n_iter=10)
#
#    print("Final result:", optimizer.max)
    
def optimize_xgb(data, targets):
    def xgb_crossval(max_depth, gamma, colsample_bytree):
        return xgb_cv(
            max_depth=int(max_depth),
            colsample_bytree=colsample_bytree,
            
            gamma=gamma,
            #eta=0.1,
          
            data=data,
            targets=targets,
        )
    optimizer = BayesianOptimization(
            f=xgb_crossval,
            pbounds={
                    "colsample_bytree":(0.3, 0.9),
                    "max_depth":(3, 7),
                    "gamma":(0, 1)
                    
                    },
            random_state=1234,
            verbose=2
    )
    optimizer.maximize(n_iter=10)
    print("Final result:", optimizer.max)
    #Final result: {'target': 0.9609421000981355, 'params': {'colsample_bytree': 0.5817010509438452, 'gamma': 0.057296734212890055, 'max_depth': 6.9169069991551515}}


#def optimize_rfc(data, targets):
#    """Apply Bayesian Optimization to Random Forest parameters."""
#    def rfc_crossval(n_estimators, min_samples_split, max_features):
#        """Wrapper of RandomForest cross validation.
#        Notice how we ensure n_estimators and min_samples_split are casted
#        to integer before we pass them along. Moreover, to avoid max_features
#        taking values outside the (0, 1) range, we also ensure it is capped
#        accordingly.
#        """
#        return rfc_cv(
#            n_estimators=int(n_estimators),
#            min_samples_split=int(min_samples_split),
#            max_features=max(min(max_features, 0.999), 1e-3),
#            data=data,
#            targets=targets,
#        )
#
#    optimizer = BayesianOptimization(
#        f=rfc_crossval,
#        pbounds={
#            "n_estimators": (10, 250),
#            "min_samples_split": (2, 25),
#            "max_features": (0.1, 0.999),
#        },
#        random_state=1234,
#        verbose=2
#    )
#    optimizer.maximize(n_iter=10)

    #print("Final result:", optimizer.max)
if __name__ == "__main__":
    
#    print(Colours.yellow("--- Optimizing SVM ---"))
#    optimize_svc(data, targets)
#    print(Colours.green("--- Optimizing Random Forest ---"))
#    
#    optimize_rfc(data, targets)
    print(Colours.green("--- XGboost ---"))
    optimize_xgb(data, targets)
    # Final result: {'target': -0.2464449012248331, 'params': {'colsample_bytree': 0.3023814615005768, 'gamma': 0.9941908961396094, 'max_depth': 3.105523507289568}}
