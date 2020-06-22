# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:12:23 2019

@author: Dell
"""

#import pandas
#import numpy as np
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
#from sklearn import cross_validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
#from sklearn import svm
import scipy.io
#from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression
mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/PAMAP_Features.mat')
data=mat['PAMAP_Features']
X=data[:,0:684]
Y=data[:,684]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.30,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#from sklearn .discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda=LDA(n_components=None)
#X=lda.fit_transform(X,Y)
#from sklearn .discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda=LDA(n_components=None)
#X_train=lda.fit_transform(X_train,Y_train)
#X_test=lda.transform(X_test)
#from sklearn.decomposition import PCA
#pca=PCA(n_components=None)
#X_train=pca.fit_transform(X_train)
#X_test=pca.transform(X_test)
#explained_Variance=pca.explained_variance_ratio_
#from sklearn.decomposition import KernelPCA
#kpca=KernelPCA(n_components=None,kernel='rbf')
#X_train=kpca.fit_transform(X_train)
#X_test=kpca.transform(X_test)
#np.random.seed(123)
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3= svm.SVC(gamma='scale', probability=True)
#cl4=XGBClassifier()
clf4=GradientBoostingClassifier()

print('10-fold cross validation:\n')

#np.random.seed(123)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3,clf4], weights=[1,1,1,1],voting='soft')
#from sklearn.model_selection import ShuffleSplit
#for clf, label in zip([clf1, clf2, clf3], ['Logistic Regression', 'Random Forest', 'SVM']
#for clf, label in zip([clf1, clf3, cl4,eclf], ['Logistic Regression','RandomForest','SVM','Xgboost','Voting Ensemble']):
    
    
    
#    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
#    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std(), label))
eclf.fit(X_train,Y_train)
y_pred=eclf.predict(X_test)
print(accuracy_score(Y_test,y_pred)*100)
X= np.concatenate((X_train,X_test), 0)
Y=np.concatenate((Y_train,Y_test), 0)

    
#    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
#    scores=cross_val_score(clf, X, y, cv=cv)
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#    accuracies=cross_val_score(estimator=clf,X=X,y=Y,cv=10)
#    print(accuracies.mean()*100,accuracies.std()*100)
    
#    print("Accuracy: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
Mamun_confusion_matrix=confusion_matrix(Y_test, y_pred,labels=[1,2,3,4,5,6,12,13])
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
print('Overall accuracy:',ACC*100)
print('False negative rate:',FNR*100)
print('false positive rate:',FPR*100)
from sklearn.metrics import classification_report
report = classification_report(Y_test, y_pred)
print(report)
print('10-fold cross validation:\n')
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(eclf,X=X,y=Y,cv=10)
print(accuracies.mean()*100,accuracies.std()*100)
Mamun_confusion_matrix=confusion_matrix(Y_test, y_pred,labels=[1,2,3,4,5,6,12,13])
   
