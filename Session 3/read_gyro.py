# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:12:08 2020

@author: Dell
"""
import pandas as pd
import numpy as np

ACC_data_1 = pd.read_csv('data_1602_gyro_watch.txt',sep=",",header=None)
ACC_data_1.iloc[:,5]=ACC_data_1.iloc[:,5].str.replace(";","")
w=ACC_data_1
w=w.replace(['A'],4)
w=w.replace(['D'],2)
w=w.replace(['E'],3)
U1_gyro_walking=w[w.iloc[:,1] == 4]
#U1_gyro_walking.drop(U1_gyro_walking.index[[3601]],inplace=True)
U1_gyro_sitting=w[w.iloc[:,1] == 2]
U1_gyro_standing=w[w.iloc[:,1] == 3]
U3_gyro_walking=U1_gyro_walking.to_numpy()
U3_gyro_sitting=U1_gyro_sitting.to_numpy()
U3_gyro_standing=U1_gyro_standing.to_numpy()



