# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:25:32 2020

@author: Dell
"""

from PyEMD import EMD
import numpy as np
from PyEMD import CEEMDAN
import scipy.io

from PyEMD import EMD, Visualisation
mat = scipy.io.loadmat('F:/Masters/Thesis/Dataset/crossposition-activity-recognition/Dataset_PerCom18_STL/Dataset_PerCom18_STL/LOSO Feature/raw_data_user_1.mat')
data=mat['raw_data_user_1']
x=data[0:500:,0]
# Extract imfs and residue
# In case of EMD
t=np.arange(0, 5, 0.01)
emd = EMD()
emd.emd(x)
imfs, res = emd.get_imfs_and_residue()
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()


