from flask import Flask, request
from flask_cors import CORS
import os
from pypylon import pylon
import cv2
import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import matplotlib.cm as cm
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import warnings
warnings.filterwarnings('ignore')
import time
import Hole_004_SIFT_Hmatrix_003 as Hole_004
import Hole_006_RegionImage_dst_002 as Hole_006
import Hole_007_LoadNpArray_64x64_003 as Hole_007
import Hole_008_PredictionFirstTwo_006 as Hole_008


app = Flask(__name__)
CORS(app)

@app.route('/dir', methods=['GET'])
def get_directories():
    current_directory = os.getcwd()
    directories = os.listdir(f'../result/models')
    return {'dir': directories}


@app.route('/pred', methods=['GET'])
def get_prediction():
    # delete all files in directory: images/partno1/pred/*.*
    prod_name = request.args.get('value1')
    os.system('del /S /Q .\\images\\partno1\\pred\\*.*')
    print('get_sample')
    return predict(prod_name)
    # return {'prediction': [
    #     {'position': 'N', 'fail_count': 3},
    #     {'position': 'E', 'fail_count': 1},
    #     {'position': 'S', 'fail_count': 0},
    #     {'position': 'W', 'fail_count': 2},
    # ]}

@app.route('/sample', methods=['GET'])
def get_sample():
    # delete all files in directory: images/partno1/pred/*.*
    os.system('del /S /Q .\\images\\partno1\\sample\\*.*')
    device_info = {}
    device_info['2676017D53E2'] = 'E'
    device_info['2676017E15AE'] = 'S'
    device_info['2676017E189A'] = 'N'
    device_info['2676017D0FF9'] = 'W'
    instance = pylon.TlFactory.GetInstance()
    print(instance)
    # Get all the available devices
    devices = instance.EnumerateDevices()

    for i, device in enumerate(devices):
        # 創建相機對象
        # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        print(f'Create camera {i}')
        camera = pylon.InstantCamera(instance.CreateDevice(devices[i]))
        guid = camera.GetDeviceInfo().GetDeviceGUID()
        print(f'Create guid {guid} = position: {device_info[guid]}')
        # 打開相機
        camera.Open()

        # 調整相機曝光時間
        camera.ExposureAuto.SetValue('Off')
        camera.ExposureTime.SetValue(100000)

        # 拍照存檔
        camera.StartGrabbing()
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = grabResult.Array
            img_path = f'..\\images\\partno1\\sample\\{device_info[guid]}.BMP'
            cv2.imwrite(img_path, image)
            print(img_path)
        grabResult.Release()
        camera.StopGrabbing()

    return { 'success': True }

    
 
def predict(prod_name): 
    all_start_time = time.time()
    directions = ['S', 'W', 'E', 'N']
    counts = []
    # directions = ['W']
    for direction in directions:
        # 訓練用照片
        input_source = '../trainning/image_demo/'+prod_name+'_'+direction+'1.BMP'
        # 測試用照片
        # input_target = '../images/partno1/sample/'+prod_name+'_'+direction+'.BMP'
        input_target = '../images/partno1/sample/'+direction+'.BMP'
        
        all_location = Hole_004.main(input_source, input_target, direction, prod_name)
        Hole_006.main(direction, prod_name)
        
        X_train, X_test, X_train_2 = Hole_007.main(direction, prod_name)
        temp_time = time.time()
        
        temp_execution_time = temp_time - all_start_time

        print(f"Program execution temp time is: {temp_execution_time} seconds\n")
        counts.append(Hole_008.main(direction, X_train, X_test, prod_name, input_target, all_location))
        
    all_end_time = time.time()
    
    all_execution_time = all_end_time - all_start_time

    print(f"Program execution total time is: {all_execution_time} seconds\n")
    
    return{'prediction': [
        {'position': 'N', 'fail_count': counts[3]},
        {'position': 'E', 'fail_count': counts[2]},
        {'position': 'S', 'fail_count': counts[0]},
        {'position': 'W', 'fail_count': counts[1]},
    ]}






    





if __name__ == '__main__':
    app.run(host='localhost', port=5000)
