# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:27:49 2020

@author: Dell
"""


import timeit

import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
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
from enn.enn import ENN
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
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import itertools    
from sklearn.model_selection import ShuffleSplit

#X=sc.fit_transform(X)
#from sklearn.model_selection import cross_val_score
#accuracies=cross_val_score(classifier,X=X,y=Y,cv=10)
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
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
data.drop(data[data.iloc[:,702] ==1].index, inplace = True)
data.drop(data[data.iloc[:,702] ==5].index, inplace = True)
data.drop(data[data.iloc[:,702] ==6].index, inplace = True)
data.drop(data[data.iloc[:,702] ==7].index, inplace = True)
data.drop(data[data.iloc[:,702] ==8].index, inplace = True)


from sklearn.utils import shuffle
data=shuffle(data)
X=data.iloc[:,0:156]
# X=X.iloc[:,m]
# X=X.iloc[:,0:26]
Y=data.iloc[:,702]
Y=Y.astype(np.float64)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
#classifier=svm.SVC(probability=True,gamma='scale')
#classifier=RandomForestClassifier()
#classifier=XGBClassifier()
#classifier=svm.SVC( C=12,gamma=0.0006, probability=True)
#classifier=svm.SVC( C= 1.8345947594465675,gamma=0.1, probability=True)
#classifier=KNeighborsClassifier()
clf1=svm.SVC(probability=True,gamma='scale');
clf2=RandomForestClassifier(n_estimators=250);
clf3=XGBClassifier();
clf4 =GradientBoostingClassifier()
clf5=KNeighborsClassifier()

classifier =EnsembleVoteClassifier(clfs=[clf2, clf3,clf1,clf4,clf5 ],weights=[1,1,1,1,1],voting='soft')
#classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('k',clf4)],voting='hard')
for clf, label in zip([clf1, clf2, clf3,clf4,clf5,classifier ], ['SVM', 'Random Forest', 'XGBoost','Gradient Boosting','k-nearest neighbour','Voting Classifier']):
    clf.fit(X,Y)
    us1=pd.read_csv('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/wisdm-dataset/wisdm-dataset/Test_4_user/Feature_U3.csv')
   
    X_test=us1.iloc[:,0:156]

    X_test=sc.fit_transform(X_test)
    Y_test=us1.iloc[:,156]                  #change
    y_pred=clf.predict(X_test)
    print("Accuracy/F1 score of the ",label)

    from sklearn.metrics import f1_score
   
    print(accuracy_score(Y_test,y_pred)*100,'/',f1_score(Y_test,y_pred,average='macro')*100)
    


# classifier.fit(X,Y)
# us1=pd.read_csv('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/wisdm-dataset/wisdm-dataset/Test_4_user/Feature_U1.csv')
# X_test=us1.iloc[:,0:156]

# X_test=sc.fit_transform(X_test)
# Y_test=us1.iloc[:,156]                  #change
# y_pred=classifier.predict(X_test)
# print("Accuracy/F1 score of the  model")

# from sklearn.metrics import f1_score
# print(accuracy_score(Y_test,y_pred)*100,'/',f1_score(Y_test,y_pred,average='macro')*100)