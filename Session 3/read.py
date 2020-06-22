# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:12:08 2020

@author: Dell
"""
import pandas as pd
import numpy as np

ACC_data_1 = pd.read_csv('data_1602_accel_watch.txt',sep=",",header=None)
ACC_data_1.iloc[:,5]=ACC_data_1.iloc[:,5].str.replace(";","")
w=ACC_data_1
w=w.replace(['A'],4)
w=w.replace(['D'],2)
w=w.replace(['E'],3)
U1_acc_walking=w[w.iloc[:,1] == 4]

#U1_acc_walking.drop(U1_acc_walking.index[[3604,3603]],inplace=True)
U1_acc_sitting=w[w.iloc[:,1] == 2]
#U1_acc_sitting.drop(U1_acc_sitting.index[[3604,3603]],inplace=True)

U1_acc_standing=w[w.iloc[:,1] == 3]
#U1_acc_standing.drop(U1_acc_standing.index[[3700,3699]],inplace=True)
U1_acc_walking.to_csv(r'F:\Masters\Thesis\Dataset\crossposition-activity-recognition\Dataset_PerCom18_STL\Dataset_PerCom18_STL\wisdm-dataset\wisdm-dataset\raw\watch\accel\U1_acc_walking.csv',index=False,header=False)

U3_acc_walking=U1_acc_walking.to_numpy()
# U3_acc_sitting=U1_acc_sitting.to_numpy()
# U3_acc_standing=U1_acc_standing.to_numpy()
# from numpy import asarray
# from numpy import savetxt
# np.load('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/wisdm-dataset/wisdm-dataset/User1 feature/Accelerometer/U1_acc_sitting.npy', allow_pickle=True)
# data = asarray(c)
# savetxt('data.csv', data,delimiter=',',fmt='%s')


