# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 03:01:05 2020

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 02:18:48 2019

@author: Dell
"""
s=[4, 5, 6, 7, 59, 64, 69, 74, 75, 84, 85, 98, 137, 147, 153, 161, 162, 163, 165, 178, 185, 197, 233, 238, 240]

m=[ 251,   319,   549 ,   31 ,  409 ,  237,   270,   548,   561,    21 ,  277,   405,   346,    9,   555,    86,   265,   502,    84,   343,547,  513,   249,   281,   415,   327,   494,    49,   255,   404,   560,   354,    71,   593,   283,   408,   498,    37,   553,   349, 257,   483,   432,   339,   546,   577,    33,   356,   510,   488]
f=[237,   561,    44,    86,   409,   398,   319,   669,   318,   474,    84,    60,   302,    85,    66,     6,   649,   528,   338,   293, 375,   371,   104,   698,    98,   200]
import timeit
from enn.enn import ENN

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
 #label_binarize
from collections import Counter

from imblearn.under_sampling import OneSidedSelection

from numpy import  where
from imblearn.under_sampling import EditedNearestNeighbours


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#conda install --user mlxtend
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions


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

mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/Feature_User_all_tsne.mat')
data=mat['Feature_User_all_tsne']
#mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/PAMAP_Features.mat')
#data=mat['PAMAP_Features']
#dataset=pandas.read_csv('pamap_2.csv')
#X=data.iloc[:, 0:684]

from sklearn.utils import shuffle
#data=shuffle(data)

# mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/Feature_tsne_excluding_user_5.mat')
# data=mat['c']
#dataset=pandas.read_csv('pamap_2.csv')
#X=data.iloc[:, 0:684]
X=data[:,0:702]
#Y=data[:,684]
#X=data[:,156:234]
X=X[:,f]
# X=X[:,0:26]
Y=data[:,702]
Y=Y.astype(np.float64)
#undersample=OneSidedSelection(n_neighbors=1,n_seeds_S=3)
undersample=EditedNearestNeighbours(n_neighbors=3)

#X,Y=undersample.fit_resample(X,Y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.30,random_state=0,stratify=Y)

#Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#X=sc.fit_transform(X)
#X_train=sc.fit_transform(X_train)
#X_test=sc.transform(X_test)
#X_mrmr=sc.fit_transform(X)

#from sklearn.preprocessing import MinMaxScaler
#sc_minmax=MinMaxScaler()
#X_train=sc_minmax.fit_transform(X_train)
#X_test=sc_minmax.transform(X_test)
#from sklearn.preprocessing import RobustScaler
#sc_robust=RobustScaler()
##X_train=sc_robust.fit_transform(X_train)
##X_test=sc_robust.transform(X_test)



#applying PCA
#from sklearn.decomposition import PCA
#pca=PCA(n_components=None)
#X_train=pca.fit_transform(X_train)
#X_test=pca.transform(X_test)
#explained_Variance=pca.explained_variance_ratio_
##applying KPCA
#from sklearn.decomposition import KernelPCA
#kpca=KernelPCA(n_components=27,kernel='rbf')
#X_train=kpca.fit_transform(X_train)
#X_test=kpca.transform(X_test)
#explained_Variance=kpca.explained_variance_ratio_
# applying LDA
#from sklearn .discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda=LDA(n_components=None)
#X_train=lda.fit_transform(X_train,Y_train)
#X_test=lda.transform(X_test)
#classifier=svm.SVC(probability=True,gamma='scale')
#classifier=AdaBoostClassifier(base_estimator=svm.SVC(probability=True,gamma='scale'))
#classifier=KNeighborsClassifier()

#classifier=GradientBoostingClassifier()
#classifier=RandomForestClassifier()
#classifier=XGBClassifier()    
classifier=ENN()      
         
        # l rate 0.05 silo gamma=0.005933508022491596
#classifier=LogisticRegression()
#############################################################################################
                             #Best parameter Tuning
#hyperopt random forest
#classifier=RandomForestClassifier(criterion="gini", max_depth=19, max_features= 23,n_estimators=97,max_leaf_nodes=174)
#hyperopt XGB
#classifier=XGBClassifier(colsample_bytree=0.5695172253161522,gamma=0.31923627844868496,max_depth=17,n_estimators=86,learning_rate=0.08,n_jobs=4,)# l rate 0.05 silo gamma=0.005933508022491596
#Bayesoptopt SVM
#classifier=svm.SVC(C=12,gamma=0.0006, probability=True)
####################################################################################################
#############################################################################################
                             #Best parameter Tuning_mrmr
#hyperopt random forest
#classifier=RandomForestClassifier(criterion="entropy", max_depth=26, max_features= 7,n_estimators= 35)
#hyperopt XGB
#classifier=XGBClassifier(colsample_bytree= 0.835273845118115,gamma=0.32674008188006337,max_depth=10,n_estimators=80,learning_rate=0.08,n_jobs=4,)# l rate 0.05 silo gamma=0.005933508022491596
#Bayesoptopt SVM
#classifier=svm.SVC( C= 1.8345947594465675,gamma=0.1, probability=True)
####################################################################################################

 #Best parameter Tuning_fcbf
#hyperopt random forest
#classifier=RandomForestClassifier(criterion="entropy", max_depth=32, max_features= 6,n_estimators= 73)
#hyperopt XGB
#classifier=XGBClassifier(colsample_bytree= 0.30668370093272074,gamma=0.21091926703991193,max_depth=12,n_estimators=66,learning_rate=0.08,n_jobs=4,)# l rate 0.05 silo gamma=0.005933508022491596
#Bayesoptopt SVM
#classifier=svm.SVC( C=10,gamma=0.010911796641235458, probability=True)
####################################################################################################
tic=timeit.default_timer()
classifier.fit(X_train,Y_train)
toc=timeit.default_timer()
print( " train time",toc - tic )
#print("Model has been trained!")
tic=timeit.default_timer()
y_pred=classifier.predict(X_test)
toc=timeit.default_timer()
print( " Predicting time",toc - tic )
print("Accuracy/F1 score of the  model")

from sklearn.metrics import f1_score

#print(f1_score(Y_test,y_pred,average='macro'))
print(accuracy_score(Y_test,y_pred)*100,'/',f1_score(Y_test,y_pred,average='macro')*100)
#Mamun_confusion_matrix=confusion_matrix(Y_test, y_pred,labels=[1,2,3,4,5,6,12,13])

#clf_svm=svm.SVC(gamma='scale')
#clf_svm.fit(X_train,Y_train)
#y_pred_svm=clf_svm.predict(X_test)
#classifier=svm.SVC( C=2.714034048174758,gamma=00.00088025910628894, probability=True)
###classifier=svm.SVC( C=7.9410702976681735,gamma=0.0006573214670265969, probability=True)
#classifier=svm.SVC(random_state=0, probability=True)

###learning curve for error metrics
#X_test=X_test[0:50:,]
                             ######LOSO model############
# matu = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/Feature_user_5.mat')
# data1=matu['Feature_user_5']  
# user_5=data1[:,m]
# #user_5=user_5[:,0:25]
# user_5=sc.fit_transform(user_5)
# Y_test=data1[:,702]                           
# classifier.fit(X,Y)
# y_pred=classifier.predict(user_5)

# print(accuracy_score(Y_test,y_pred)*100)
# plt.figure(dpi=100)
# plot_learning_curves(X_train, Y_train, X_test, Y_test, classifier, print_model=False, style='ggplot',scoring='accuracy',cv=10)
# plt.show()

#print("Accuracy of  SVM")
#print(accuracy_score(Y_test,y_pred_svm)) 
#Mamun_confusion_matrix=confusion_matrix(Y_test, y_pred_svm,labels=[1,2,3,4,5])
#10 fold cross validaton
#X=sc.fit_transform(X)
X1= np.concatenate((X_train,X_test), 0)
Y1=np.concatenate((Y_train,Y_test), 0)


#accuracies=cross_val_score(classifier,X=X1,y=Y1,cv=10)
from sklearn.metrics import accuracy_score
accuracies=cross_val_score(classifier,X=X1,y=Y1,cv=10,scoring='accuracy')
F1_score=cross_val_score(classifier,X=X1,y=Y1,cv=10,scoring='f1_macro')
from sklearn.metrics import accuracy_score
print("K fold accuracy")

print(accuracies.mean()*100,accuracies.std()*100)
print("K fold F1 score")
print(F1_score.mean()*100,F1_score.std()*100)

#cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
#scores=cross_val_score(classifier, X, Y, cv=cv)


#def perf_measure(Y_test, y_pred):
#    TP = 0
#    FP = 0
#    TN = 0
#    FN = 0
#
#    for i in range(len(y_pred)): 
#        if Y_test[i]==y_pred[i]==1:
#           TP += 1
#        if y_pred[i]==1 and Y_test[i]!=y_pred[i]:
#           FP += 1
#        if Y_test[i]==y_pred[i]==0:
#           TN += 1
#        if y_pred[i]==0 and Y_test[i]!=y_pred[i]:
#           FN += 1
#           
#    
#
#    return(TP, FP, TN, FN)
#print(perf_measure(Y_test, y_pred))
Mamun_confusion_matrix=confusion_matrix(Y_test, y_pred,labels=[1,2,3,4,5,6,7,8])
confusion_matrix=Mamun_confusion_matrix;
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)
TP=np.sum(TP)

TN=np.sum(TN)

FP=np.sum(FP)

FN=np.sum(FN)

 #Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
F1=2*((TPR*PPV)/(TPR+PPV))

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("         PERFORMANCE METRICS     ")
print('Recall:',TPR*100)  
print('Specificity:',TNR*100)
print('Precision:',PPV*100)
print('False discovery rate:',FDR*100)
print('F1 score:',F1*100)
#print('Overall accuracy:',ACC*100)
print('False negative rate:',FNR*100)
print('false positive rate:',FPR*100)

report = classification_report(Y_test, y_pred)
print(report)
#roc score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    
    lb = LabelBinarizer()
    lb.fit(Y_test)
    
    y_test = lb.transform(Y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
print(multiclass_roc_auc_score(Y_test,y_pred))

##SFSS
# feat_cols = list(sfs1.k_feature_idx_)
#print(feat_cols)
#c=[4, 5, 6, 7, 59, 64, 69, 74, 75, 84, 85, 98, 137, 147, 153, 161, 162, 163, 165, 178, 185, 197, 233, 238, 240]
#X=X[:, feat_cols]
#mrmr
#d=[ 252,   320,   550 ,   32 ,  410 ,  238,   271,   549,   562,    22 ,  278,   406,   347,    10,   556,    87,   266,   503,    85,   344,548,  514,   250,   282,   416,   328,   495,    50,   256,   405,   561,   355,    72,   594,   284,   409,   499,    38,   554,   350, 258,   484,   433,   340,   547,   578,    34,   357,   511,   489]
#fcbf
#f=[237,   561,    44,    86,   409,   398,   319,   669,   318,   474,    84,    60,   302,    85,    66,     6,   649,   528,   338,   293, 375,   371,   104,   698,    98,   200]
#y=Y.flatten();y.astype(np.integer);