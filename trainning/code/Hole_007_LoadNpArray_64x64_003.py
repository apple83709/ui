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
def main(direction, prod_name):
    FileLoc = '../../result/06_SortedTrainTest/'+prod_name
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
    
    
    # direction = 'N'
    numpy_src = np.empty((0,32,32))
    for loop in range (1,7):
        
        file_src = '../../result/05_RegionImage/'+prod_name+'/np-src-'+direction+'-'+str(loop) + '.npy'
        tmp_src = np.load(file_src)
        # Append the array vertically
        numpy_src = np.vstack((numpy_src, tmp_src))
        
    
    
    
    
    numpy_dst = np.empty((0,32,32))
    for idx in range (1,7):
    
        file_dst = '../../result/05_RegionImage/'+prod_name+'/np-dst-'+direction+'-'+str(idx) + '.npy'
        tmp_dst = np.load(file_dst)
        # Append the array vertically
        numpy_dst = np.vstack((numpy_dst, tmp_dst))
    
    
    # =============================================================================
    # 
    # convert the train data
    # 
    # =============================================================================
    
    if(len(numpy_src) % 9 != 0):
        X_train = np.zeros((int(len(numpy_src)/9)+1,3,224,224)).astype(np.uint8)
    else:
        X_train = np.zeros((int(len(numpy_src)/9),3,224,224)).astype(np.uint8)
    
    x64s = []
    for i in range (0,len(numpy_src)):
    # for i in range (0,1):
    
        x_tmp = numpy_src[i,:,:]
        x32 = np.stack([x_tmp] * 3, axis=0)
        x64 = np.repeat(np.repeat(x32, 2, axis=1), 2, axis=2)
        x64s.append(x64)
    

    for idx in range(0, int(len(x64s)/9)):
        r2 = idx * 9
        X_train[idx,:,13:13+64,13:13+64] = x64s[r2]
        X_train[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+1]
        X_train[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+2]
        
        
        X_train[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+3]
        X_train[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+4]
        X_train[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+5]
        
    
        X_train[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+6]
        X_train[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+7]
        X_train[idx,:,13+2*64:13+3*64,13+2*64:13+3*64] = x64s[r2+8]
    
    if(len(x64s) % 9 != 0):
        # 剩幾張
        tmp = len(x64s) - int(len(x64s)/9) * 9
        # print(tmp)
        for idx in range(int(len(x64s)/9), int(len(x64s)/9)+1):
            r2 = idx * 9
            X_train[idx,:,13:13+64,13:13+64] = x64s[r2]
            if(tmp>=2): 
                X_train[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+1]
            else:
                X_train[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+tmp-1]
            if(tmp>=3): 
                X_train[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+2]
            else:
                X_train[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+tmp-1]
            
            if(tmp>=4): 
                X_train[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+3]
            else:
                X_train[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+tmp-1]
            if(tmp>=5): 
                X_train[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+4]
            else:
                X_train[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+tmp-1]
            if(tmp>=6): 
                X_train[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+5]
            else:
                X_train[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+tmp-1]
            
            
            if(tmp>=7): 
                X_train[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+6]
            else:
                X_train[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+tmp-1]
            if(tmp>=8):
                X_train[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+7]
            else:
                X_train[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+tmp-1]
                
            X_train[idx,:,13+2*64:13+3*64,13+2*64:13+3*64] = x64s[r2+tmp-1]
            
    
    # print('123')
        
    # image_src = np.transpose( X_train[int(len(x64s)/9),:,:,:], (1, 2, 0))    
    # plt.imshow(image_src)
    # plt.show()
    
    
    # =============================================================================
    # 
    # convert the train data
    # 
    # =============================================================================
    
    if(len(numpy_dst) % 9 != 0):
        X_test = np.zeros((int(len(numpy_dst)/9)+1,3,224,224)).astype(np.uint8)
    else:
        X_test = np.zeros((int(len(numpy_dst)/9),3,224,224)).astype(np.uint8)
    
    x64s = []
    for i in range (0,len(numpy_dst)):
    
        x_tmp = numpy_dst[i,:,:]
        x32 = np.stack([x_tmp] * 3, axis=0)
        x64 = np.repeat(np.repeat(x32, 2, axis=1), 2, axis=2)
        x64s.append(x64)
    

    for idx in range(0, int(len(x64s)/9)):
        r2 = idx * 9
        X_test[idx,:,13:13+64,13:13+64] = x64s[r2]
        X_test[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+1]
        X_test[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+2]
        
        
        X_test[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+3]
        X_test[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+4]
        X_test[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+5]
        
    
        X_test[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+6]
        X_test[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+7]
        X_test[idx,:,13+2*64:13+3*64,13+2*64:13+3*64] = x64s[r2+8]
    
    if(len(x64s) % 9 != 0):
        # 剩幾張
        tmp = len(x64s) - int(len(x64s)/9) * 9
        # print(tmp)
        for idx in range(int(len(x64s)/9), int(len(x64s)/9)+1):
            r2 = idx * 9
            X_test[idx,:,13:13+64,13:13+64] = x64s[r2]
            if(tmp>=2): 
                X_test[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+1]
            else:
                X_test[idx,:,13+64:13+2*64,13:13+64] = x64s[r2+tmp-1]
            if(tmp>=3): 
                X_test[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+2]
            else:
                X_test[idx,:,13+2*64:13+3*64,13:13+64] = x64s[r2+tmp-1]
            
            if(tmp>=4): 
                X_test[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+3]
            else:
                X_test[idx,:,13:13+64,13+64:13+64*2] = x64s[r2+tmp-1]
            if(tmp>=5): 
                X_test[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+4]
            else:
                X_test[idx,:,13+64:13+2*64,13+64:13+64*2] = x64s[r2+tmp-1]
            if(tmp>=6): 
                X_test[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+5]
            else:
                X_test[idx,:,13+2*64:13+3*64,13+64:13+64*2] = x64s[r2+tmp-1]
            
            
            if(tmp>=7): 
                X_test[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+6]
            else:
                X_test[idx,:,13:13+64,13+2*64:13+3*64] = x64s[r2+tmp-1]
            if(tmp>=8):
                X_test[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+7]
            else:
                X_test[idx,:,13+64:13+2*64,13+2*64:13+3*64] = x64s[r2+tmp-1]
                
            X_test[idx,:,13+2*64:13+3*64,13+2*64:13+3*64] = x64s[r2+tmp-1]
        
    # image_dst = np.transpose( X_test[ int(len(x64s)/9),:,:,:], (1, 2, 0))    
    # plt.imshow(image_dst)
    # plt.show()
    
    
    # print('456')
    X_train_2 = np.zeros((len(numpy_src),3,224,224)).astype(np.uint8)


    for i in range (0,len(numpy_src)):
    # for i in range (0,1):
        x_tmp = numpy_src[i,:,:]
        x32 = np.stack([x_tmp] * 3, axis=0)
        x64 = np.repeat(np.repeat(x32, 2, axis=1), 2, axis=2)
        
        X_train_2[i,:,13:13+64,13:13+64] = x64
        X_train_2[i,:,13+64:13+2*64,13:13+64] = x64
        X_train_2[i,:,13+2*64:13+3*64,13:13+64] = x64
        
        
        X_train_2[i,:,13:13+64,13+64:13+64*2] = x64
        X_train_2[i,:,13+64:13+2*64,13+64:13+64*2] = x64
        X_train_2[i,:,13+2*64:13+3*64,13+64:13+64*2] = x64
        

        X_train_2[i,:,13:13+64,13+2*64:13+3*64] = x64
        X_train_2[i,:,13+64:13+2*64,13+2*64:13+3*64] = x64
        X_train_2[i,:,13+2*64:13+3*64,13+2*64:13+3*64] = x64
    
    # image_src2 = np.transpose( X_train_2[0,:,:,:], (1, 2, 0))    
    # plt.imshow(image_src2)
    # plt.show()
    
    
    # print('789')    
    
    # =============================================================================
    # 
    # save the X_train and X_test
    # 
    # =============================================================================
    return X_train, X_test, X_train_2
    # out_X_train_file = '../result/06_SortedTrainTest/X_train_'+direction+'.npy'
    # np.save(out_X_train_file, X_train)
    
    # out_X_test_file = '../result/06_SortedTrainTest/X_test_'+direction+'.npy'
    # np.save(out_X_test_file, X_test)
    
    
X_train, X_test, X_train_2 = main('W', '2024-09-04')
    
