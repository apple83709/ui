# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""

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

def main(direction, X_train, X_test, X_train_2, all_location, image_target, prod_name):
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = 'cpu'
    print('device =', device)
    
    
    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    
        return z
    
    
    
    def denormalization(x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        
        return x
    
    ####################################################################
    #
    # load the Resnet-18 model:
    #
    ####################################################################    
    
    
    # train_feature_filepath = './train_feature_20240819.pkl'
    
    
    # load model
    
    model = resnet18(pretrained=True, progress=True)
    t_d = 192
    d = 50
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    
    
    if use_cuda:
        torch.cuda.manual_seed_all(1024)
    
    idx = torch.tensor(sample(range(0, t_d), d))
    
    # set model's intermediate outputs
    outputs = []
    
    
    
    def hook(module, input, output):
        outputs.append(output)
        
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    
    ArithmeticErrortotal_roc_auc = []
    total_pixel_roc_auc = []
    
    ####################################################################
    #
    # train the normal data
    #
    ####################################################################    
    
    
    # Define the dataset class
    class MyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    
    # =============================================================================
    # 
    # load the X_tain and X_test files
    # 
    # =============================================================================
    # direction = 'N'
    # for loop in range(1, 7):
    
    file_name1 = '../../result/idxs/'+prod_name+'/idx_'+direction+'.pkl'
    with open(file_name1, 'wb') as f:
        pickle.dump(idx, f)
        
    
    # out_X_train_file = '../result/06_SortedTrainTest/X_train_'+direction+'.npy'
    # np_1 = np.load(out_X_train_file)
    np_1 = X_train
    train_image_in_numpy = np_1.astype(float)/255.0
    print('np_1 size:' + str(np_1.shape))
    
    np_3 = X_train_2
    train_image_in_numpy_1 = np_3.astype(float)/255.0
    print('np_3 size:' + str(np_3.shape))
    
    # out_X_test_file = '../result/06_SortedTrainTest/X_test_'+direction+'.npy'
    # np_2 = np.load(out_X_test_file)
    np_2 = X_test
    print('np_2 size:' + str(np_2.shape))
    test_image_in_numpy = np_2.astype(float)/255.0
    
    
    # test_image_in_numpy = np.random.rand(420,3,224,224)
    # train_image_in_numpy = np.random.rand(420,3,224,224)
    
    
    # =============================================================================
    # 
    # prepare input data
    # 
    # =============================================================================
    
    
    train_target_in_numpy = np.ones((len(train_image_in_numpy), 1), dtype=np.uint8)
    train_target_in_numpy_2 = np.ones((len(train_image_in_numpy_1), 1), dtype=np.uint8)
    test_target_in_numpy = np.ones((len(test_image_in_numpy), 1), dtype=np.uint8)
    
    

    # =============================================================================
    # 
    # Prepare dataloader for train images
    # 
    # =============================================================================
    
    # Create the dataset
    dataset = MyDataset(torch.from_numpy(train_image_in_numpy).float(), torch.from_numpy(train_target_in_numpy).float())
    dataset2 = MyDataset(torch.from_numpy(train_image_in_numpy_1).float(), torch.from_numpy(train_target_in_numpy_2).float())
    # Create the DataLoader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    # *9
    train_dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False, pin_memory=True)
    
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    cov_train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    
    
    
    flag_train = 1
    
    if (flag_train ==1):
        for (x, _) in tqdm(train_dataloader):
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        # for layer_name in ['layer2', 'layer3']:
        #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
    
        for layer_name in ['layer2']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
    
    
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        print("embedding_vectors.size() = ", embedding_vectors.size())
        
        # embedding_vectors_reduced = embedding_vectors[:,:,4:52,4:52]
        embedding_vectors_reduced = embedding_vectors.clone()
        print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
        
     
        embedding_vectors_reduced = embedding_vectors_reduced.reshape(B, C, 56*56)
        print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
        # embedding_vectors.size() =  torch.Size([209, 100, 3136])
       
        embedding_vectors_reduced_train = embedding_vectors_reduced.clone().numpy()
        with open('../../result/embedding_vectors_reduced_trains/'+prod_name+'/embedding_vectors_reduced_train_'+direction+'.npy', 'wb') as f:
            pickle.dump(embedding_vectors_reduced_train, f)
        
        mean = torch.mean(embedding_vectors_reduced, dim=0).numpy()
        
        mean_train = mean.copy()
        
        
        cov = torch.zeros(C, C, 56*56).numpy()
        I = np.identity(C)
        for i in range( 56*56):
            cov[:, :, i] = np.cov(embedding_vectors_reduced[:, :, i].numpy(), rowvar=False) + 0.01 * I
    
        
        train_outputs_reduced = [mean, cov]
        
        # print("mean.shape, cov.shape = ", mean.shape, cov.shape)
        # mean.shape, cov.shape =  (100, 3136) (100, 100, 3136)
        # train_output is a dictionary with two elements: mean and cov
     
        # with open(train_feature_filepath, 'wb') as f:
        #     pickle.dump(train_outputs_reduced, f)
        
        torch.save(model.state_dict(), '../../result/models/'+prod_name+'/save_'+direction +'.pt')
        
        
        # *9 拿cov_inv
        for (x, _) in tqdm(train_dataloader2):
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(cov_train_outputs.keys(), outputs):
                cov_train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in cov_train_outputs.items():
            cov_train_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = cov_train_outputs['layer1']
        # for layer_name in ['layer2', 'layer3']:
        #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
    
        for layer_name in ['layer2']:
            embedding_vectors = embedding_concat(embedding_vectors, cov_train_outputs[layer_name])
    
    
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        print("embedding_vectors.size() = ", embedding_vectors.size())
        
        # embedding_vectors_reduced = embedding_vectors[:,:,4:52,4:52]
        embedding_vectors_reduced = embedding_vectors.clone()
        print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
        
     
        embedding_vectors_reduced = embedding_vectors_reduced.reshape(B, C, 56*56)
        print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
        # embedding_vectors.size() =  torch.Size([209, 100, 3136])
       
        # embedding_vectors_reduced_train = embedding_vectors_reduced.clone().numpy()
        # with open('../../result/embedding_vectors_reduced_trains/'+prod_name+'/embedding_vectors_reduced_train_'+direction+'.npy', 'wb') as f:
        #     pickle.dump(embedding_vectors_reduced_train, f)
        
        mean = torch.mean(embedding_vectors_reduced, dim=0).numpy()
        
        mean_train = mean.copy()
        
        
        cov = torch.zeros(C, C, 56*56).numpy()
        I = np.identity(C)
        for i in range( 56*56):
            cov[:, :, i] = np.cov(embedding_vectors_reduced[:, :, i].numpy(), rowvar=False) + 0.01 * I
    
        
        train_outputs_reduced = [mean, cov]
        
        print("mean.shape, cov.shape = ", mean.shape, cov.shape)
        # mean.shape, cov.shape =  (100, 3136) (100, 100, 3136)
        # train_output is a dictionary with two elements: mean and cov
     
        # with open(train_feature_filepath, 'wb') as f:
        #     pickle.dump(train_outputs_reduced, f)
        
        # torch.save(model.state_dict(), '../../result/models/'+prod_name+'/save_'+direction +'.pt')
            
    if (flag_train ==0):
        
        # print("\n#3   train_feature_filepath =", train_feature_filepath)
        # print('load train set feature from: %s' % train_feature_filepath)
        # with open(train_feature_filepath, 'rb') as f:
        #     train_outputs_reduced = pickle.load(f)
        
        embedding_vectors_reduced_train = np.load('../../result/embedding_vectors_reduced_trains/'+prod_name+'/embedding_vectors_reduced_train_'+direction+'.npy')
        # embedding_vectors_reduced_train = np.load('../../trainning/result/embedding_vectors_reduced_trains/embedding_vectors_reduced_train_'+direction+'.npy')
        weight = torch.load('../../result/models/'+prod_name+'/save_'+direction+'.pt', map_location=device)
        # weight = torch.load('save.pt')
        model.load_state_dict(weight)
        model.eval()
    
    
    ####################################################################
    #
    # predict the test dataset
    #
    ####################################################################    
    
    start_time = time.time()
    
    # =============================================================================
    # 
    # Prepare dataloader for test images
    # 
    # =============================================================================
    
    # Create the dataset
    dataset_2 = MyDataset(torch.from_numpy(test_image_in_numpy).float(), torch.from_numpy(test_target_in_numpy).float())
    
    # Create the DataLoader
    test_dataloader = DataLoader(dataset_2, batch_size=32, shuffle=False, pin_memory=True)
    
    
    gt_list = []
    gt_mask_list = []
    test_imgs = []
    
    print("\n#4   test_dataloader")
    
    index = 0        
           
    # extract test set features
    for (x, _) in tqdm(test_dataloader):
     
        test_imgs.extend(x.cpu().detach().numpy())
    
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs   
        
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        # initialize hook outputs
        outputs = []
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program execution time (first part) is: {execution_time} seconds\n")
    start_time = end_time
            
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)
        print(k)
    
        
    # =============================================================================
        
    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    # for layer_name in ['layer2', 'layer3']:
    #     embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
    
    for layer_name in ['layer2']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
    
    
    print("embedding_vectors.shape = ", embedding_vectors.shape)
    
    # embedding_vectors.shape =  torch.Size([69, 448, 56, 56])
    
    # =============================================================================
    # 
    #     in embedding_vectors:
    #     83x448x56x56
    #     
    # =============================================================================
    
    
    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    print("embedding_vectors.shape = ", embedding_vectors.shape)
    # select 50 out of 192, using idx
    # embedding_vectors.shape =  torch.Size([83, 100, 56, 56])
    
    B, C, H, W = embedding_vectors.size()
    
    # embedding_vectors_reduced = embedding_vectors[:,:,4:52,4:52]
    embedding_vectors_reduced = embedding_vectors.clone()
    print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
    
     
    embedding_vectors_reduced = embedding_vectors_reduced.reshape(B, C, 56*56)
    print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
    # embedding_vectors.size() =  torch.Size([209, 100, 3136])
       
    embedding_vectors_reduced_test = embedding_vectors_reduced.clone().numpy()
    
    
    # =============================================================================
    # 
    # calculate mahalanobis distance
    # 
    # =============================================================================
    np.save('../../result/embedding_vectors_reduced_trains/'+prod_name+'/embedding_vectors_reduced_train_'+direction+'.npy', embedding_vectors_reduced_train)
    
    
    conv_invs = []
    dist_list = []
    sample_trains = []
    print('-------embedding_vectors_reduced_train:'+ str(embedding_vectors_reduced_train[:,:,0].shape))
    print('-------embedding_vectors_reduced_test:'+ str(embedding_vectors_reduced_test[:,:,0].shape))
    
    
    for i in range(56*56):
        
        dist = []
        conv_inv = np.linalg.inv(train_outputs_reduced[1][:, :, i])
        
        conv_invs.append(conv_inv)
        sample_train = embedding_vectors_reduced_train[:,:,i]
        sample_trains.append(sample_train)
        
        sample_test  = embedding_vectors_reduced_test[:,:,i]
        sample_diff = sample_test - sample_train   
        maha_full = sample_diff @ conv_inv @ (sample_diff.T)
        
        diagonal_elements = np.diagonal(maha_full).tolist()
        dist_list.append(diagonal_elements)
    
    with open('../../result/con_invs/'+prod_name+'/conv_invs_'+direction+'.pkl', 'wb') as f:
      pickle.dump(conv_invs, f)
      
    with open('../../result/sample_trains/'+prod_name+'/sample_trains_'+direction+'.pkl', 'wb') as f:
      pickle.dump(sample_trains, f)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B,56,56)
    
    
    print("dist_list.shape",dist_list.shape)
    # (83, 56, 56)
    
    
    
    # upsample
    dist_list = torch.tensor(dist_list)
    
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()
    # score_map, Array of float32, (83,192,192) 
    
    
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        
    # score_map[19] = 0
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    
    print('max_score, min_score =', max_score, min_score)
    
    
    # max_score = 2.2
    
    # scores = score_map / max_score
    scores = score_map / max_score
    scores_min = np.minimum(scores, 1)
    
    # scores, Array of float32, (83,224,224) 
    
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Program execution time (second part) is: {execution_time} seconds\n")
    
        
    
    fail = []
    threshhold = 100/max_score
    for j in range(0, scores_min.shape[0]):
        max_value = np.max(scores_min[j])
        if max_value > threshhold:
            has_positive = True
        else:
            has_positive = False
            
        if has_positive == True:     
            for x in range(3):  
                for y in range(3):  
                    arr = scores_min[j]
                    sub_array = arr[x*75:(x+1)*75, y*75:(y+1)*75]
                    if(np.max(sub_array)>threshhold):
                        # 計算第幾個點
                        fail.append(j * 9 + (x + 1) * (y + 1))
   
    loaded_data = all_location
    all_fails=[]
    for i in range(0, len(fail)):
        all_fails.append([loaded_data['location'].iloc[fail[i]].large_x, loaded_data['location'].iloc[fail[i]].large_y])
        # print('region_' + str(loop) + '_img_loc: ', loaded_data['location'].iloc[i].large_x)
    
    # 添加紅色方塊
    # image = cv2.imread('../image_demo/2024-08-22_'+direction+'2.BMP')
    image = cv2.imread(image_target)
    for index in range(0, len(all_fails)):
        large_x = all_fails[index][0]
        large_y = all_fails[index][1]
        large_x = int(large_x)
        large_y = int(large_y)
        # 紅色
        image[large_y-16:large_y+16,large_x-16:large_x+16,2] = 255
    
    cv2.imwrite(direction+'.BMP', image)
