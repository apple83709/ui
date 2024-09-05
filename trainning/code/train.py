# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""

import Hole_002_DrawLine_003 as Hole002
import Hole_003_FindRegions_001 as Hole003
import Hole_004_SIFT_Hmatrix_003 as Hole004
import Hole_005_RegionImage_src_002 as Hole005
import Hole_006_RegionImage_dst_002 as Hole006
import Hole_007_LoadNpArray_64x64_003 as Hole007
import Hole_008_PredictionFirstTwo_004 as Hole008
import os
import time


def checkfolders(prod_name):
    FileLoc = '../../result/con_invs/'+prod_name
    try:
        os.mkdir(FileLoc)
    except OSError:
        print ("Creation of the directory %s failed" % FileLoc)
    else:
        print ("Successfully created the directory %s " % FileLoc)
        
        
    FileLoc = '../../result/sample_trains/'+prod_name
    
    try:
        os.mkdir(FileLoc)
    except OSError:
        print ("Creation of the directory %s failed" % FileLoc)
    else:
        print ("Successfully created the directory %s " % FileLoc)
        
    FileLoc = '../../result/models/'+prod_name
    
    try:
        os.mkdir(FileLoc)
    except OSError:
        print ("Creation of the directory %s failed" % FileLoc)
    else:
        print ("Successfully created the directory %s " % FileLoc)
    
    FileLoc = '../../result/idxs/'+prod_name
    
    try:
        os.mkdir(FileLoc)
    except OSError:
        print ("Creation of the directory %s failed" % FileLoc)
    else:
        print ("Successfully created the directory %s " % FileLoc)
        
    FileLoc = '../../result/embedding_vectors_reduced_trains/'+prod_name
    
    try:
        os.mkdir(FileLoc)
    except OSError:
        print ("Creation of the directory %s failed" % FileLoc)
    else:
        print ("Successfully created the directory %s " % FileLoc)
        

try:
    direction = 'E'
    prod_name = '2024-09-04'
    
    checkfolders(prod_name)
    
    
    input_file =  '../image_demo/'+prod_name+'_'+direction+'1.BMP'
    input_file_target =   '../image_demo/'+prod_name+'_'+direction+'2.BMP'
    Hole002.main(direction, input_file, prod_name)
    
    Hole003.main(direction, input_file, prod_name)   
    start_time = time.time()
    all_location = Hole004.main(direction, input_file, input_file_target, prod_name)
    Hole005.main(direction, prod_name)
    Hole006.main(direction, prod_name)
    X_train, X_test, X_train_2 = Hole007.main(direction, prod_name)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program 004~007 execution time is: {execution_time} seconds\n")
    
    Hole008.main(direction, X_train, X_test, X_train_2, all_location, input_file_target, prod_name)
    
except Exception as e:
    print(e)
    

