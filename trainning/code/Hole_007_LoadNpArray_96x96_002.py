# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""


import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import listdir
import matplotlib.pyplot as plt
import os
import warnings 
warnings.filterwarnings('ignore')
import pickle

# =============================================================================
# 
# create directory
# 
# =============================================================================

FileLoc = '../result/06_SortedTrainTest'
try:
    os.mkdir(FileLoc)
except OSError:
    print ("Creation of the directory %s failed" % FileLoc)
else:
    print ("Successfully created the directory %s " % FileLoc)
    
    
# =============================================================================
# 
# load the source and destination images and positions
# 
# =============================================================================

for loop in range(1, 7):
    file_src = '../result/05_RegionImage/np-src-'+str(loop) +'.npy'
    np_array_src = np.load(file_src)
    
    file_dst = '../result/05_RegionImage/np-dst-'+str(loop) +'.npy'
    np_array_dst = np.load(file_dst)
    
    # =============================================================================
    # 
    # convert the train data
    # 
    # =============================================================================
    
    X_train = np.zeros((len(np_array_src),3,224,224)).astype(np.uint8)
    
    
    for i in range (0,len(X_train)):
        x_tmp = np_array_src[i,:,:]
        x32 = np.stack([x_tmp] * 3, axis=0)
        x96 = np.repeat(np.repeat(x32, 3, axis=1), 3, axis=2)
        
        X_train[i,:,13:13+96,13:13+96] = x96
        X_train[i,:,13+96:13+2*96,13:13+96] = x96
        
        X_train[i,:,13:13+96,13+96:13+96*2] = x96
        X_train[i,:,13+96:13+2*96,13+96:13+96*2] = x96
        
        
    image_src = np.transpose( X_train[0,:,:,:], (1, 2, 0))    
    plt.imshow(image_src)
    plt.show()
    
    
    # =============================================================================
    # 
    # convert the train data
    # 
    # =============================================================================
    
    X_test = np.zeros((len(np_array_dst),3,224,224)).astype(np.uint8)
    
    
    for i in range (0,len(X_test)):
    
        x_tmp = np_array_dst[i,:,:]
        x32 = np.stack([x_tmp] * 3, axis=0)
        x96 = np.repeat(np.repeat(x32, 3, axis=1), 3, axis=2)
        
        X_test[i,:,13:13+96,13:13+96] = x96
        X_test[i,:,13+96:13+2*96,13:13+96] = x96
        
        X_test[i,:,13:13+96,13+96:13+96*2] = x96
        X_test[i,:,13+96:13+2*96,13+96:13+96*2] = x96
         
        
    image_dst = np.transpose( X_test[0,:,:,:], (1, 2, 0))    
    plt.imshow(image_dst)
    plt.show()
    
    
    # =============================================================================
    # 
    # save the X_train and X_test
    # 
    # =============================================================================
    
    out_X_train_file = '../result/06_SortedTrainTest/X_train_'+str(loop) +'.npy'
    np.save(out_X_train_file, X_train)
    
    out_X_test_file = '../result/06_SortedTrainTest/X_test_'+str(loop) +'.npy'
    np.save(out_X_test_file, X_test)
    print(X_test.size)
    
    
    
    
