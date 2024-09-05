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
import Hole_004_SIFT_Hmatrix_003 as Hole_004
import Hole_006_RegionImage_dst_002 as Hole_006
import Hole_007_LoadNpArray_64x64_003 as Hole_007


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device = 'cpu'
# print('device =', device)

def main(direction, X_train, X_test, prod_name, input_target, all_location):
    class MyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    outputs = [] 
    def hook(module, input, output):
        outputs.append(output)
    # train_feature_filepath = './train_feature_20240819.pkl'
    use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if use_cuda else 'cpu')
    device = 'cpu'
    # print('device =', device)

    model = resnet18(pretrained=True, progress=True)
    t_d = 192
    d = 50
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    if use_cuda:
        torch.cuda.manual_seed_all(1024)
    
    # idx = torch.tensor(sample(range(0, t_d), d))
    
    
    
    # set model's intermediate outputs
    ArithmeticErrortotal_roc_auc = []
    total_pixel_roc_auc = []
    
    all_fails = []
    
    np_1 = X_train 
    # print('np_1 size:' + str(np_1.shape))
    train_image_in_numpy = np_1.astype(float)/255.0


    np_2 = X_test
    # print('np_2 size:' + str(np_2.shape))
    test_image_in_numpy = np_2.astype(float)/255.0

    
    # test_image_in_numpy = np.random.rand(420,3,224,224)
    # train_image_in_numpy = np.random.rand(420,3,224,224)


    # =============================================================================
    # 
    # prepare input data
    # 
    # =============================================================================


    train_target_in_numpy = np.ones((len(train_image_in_numpy), 1), dtype=np.uint8)
    test_target_in_numpy = np.ones((len(test_image_in_numpy), 1), dtype=np.uint8)


    # =============================================================================
    # 
    # Prepare dataloader for train images
    # 
    # =============================================================================

    # Create the dataset
    dataset = MyDataset(torch.from_numpy(train_image_in_numpy).float(), torch.from_numpy(train_target_in_numpy).float())

    # Create the DataLoader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)


    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])


    flag_train = 0

   
    if (flag_train ==0):
        
        # # print("\n#3   train_feature_filepath =", train_feature_filepath)
        # # print('load train set feature from: %s' % train_feature_filepath)
        # with open(train_feature_filepath, 'rb') as f:
        #     train_outputs_reduced = pickle.load(f)
        
        # embedding_vectors_reduced_train = np.load('../result/embedding_vectors_reduced_trains/embedding_vectors_reduced_train_'+direction+'.npy')
        embedding_vectors_reduced_train = np.load('../result/embedding_vectors_reduced_trains/'+prod_name+'/embedding_vectors_reduced_train_'+direction+'.npy')
        weight = torch.load('../result/models/'+prod_name+'/save_'+direction+'.pt', map_location=device)
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

    # print("\n#4   test_dataloader")

    index = 0        
    # outputs = []       
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
        # print(k)

        
    # =============================================================================
        
    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    # for layer_name in ['layer2', 'layer3']:
    #     embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    for layer_name in ['layer2']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])


    # print("embedding_vectors.shape = ", embedding_vectors.shape)

    # embedding_vectors.shape =  torch.Size([69, 448, 56, 56])

    # =============================================================================
    # 
    #     in embedding_vectors:
    #     83x448x56x56
    #     
    # =============================================================================
    with open('../result/idxs/'+prod_name+'/idx_'+direction+'.pkl', 'rb') as f:
        idx = pickle.load(f)

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    # print("embedding_vectors.shape = ", embedding_vectors.shape)
    # select 50 out of 192, using idx
    # embedding_vectors.shape =  torch.Size([83, 100, 56, 56])

    B, C, H, W = embedding_vectors.size()

    # embedding_vectors_reduced = embedding_vectors[:,:,4:52,4:52]
    embedding_vectors_reduced = embedding_vectors.clone()
    # print("embedding_vectors.size() = ", embedding_vectors_reduced.size())

     
    embedding_vectors_reduced = embedding_vectors_reduced.reshape(B, C, 56*56)
    # print("embedding_vectors.size() = ", embedding_vectors_reduced.size())
    # embedding_vectors.size() =  torch.Size([209, 100, 3136])
       
    embedding_vectors_reduced_test = embedding_vectors_reduced.clone().numpy()


    # =============================================================================
    # 
    # calculate mahalanobis distance
    # 
    # =============================================================================


    dist_list = []
    # print('-------embedding_vectors_reduced_train:'+ str(embedding_vectors_reduced_train[:,:,0].shape))
    # print('-------embedding_vectors_reduced_test:'+ str(embedding_vectors_reduced_test[:,:,0].shape))

    with open('../result/con_invs/'+prod_name+'/conv_invs_'+direction+'.pkl', 'rb') as f:
        conv_invs = pickle.load(f)
    
    with open('../result/sample_trains/'+prod_name+'/sample_trains_'+direction+'.pkl', 'rb') as f:
        sample_trains = pickle.load(f)
    
    for i in range(56*56):
        
        dist = []
        conv_inv = conv_invs[i]
        sample_train = sample_trains[i]
        sample_test  = embedding_vectors_reduced_test[:,:,i]
        sample_diff = sample_test - sample_train   
        maha_full = sample_diff @ conv_inv @ (sample_diff.T)
        
        diagonal_elements = np.diagonal(maha_full).tolist()
        dist_list.append(diagonal_elements)

    # with open('../../trainning/result/con_invs/conv_invs_'+direction+'.pkl', 'wb') as f:
    #   pickle.dump(conv_invs, f)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B,56,56)


    # print("dist_list.shape",dist_list.shape)
    # (83, 56, 56)


    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Program execution time (second part) is: {execution_time} seconds\n")
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

    # print('max_score, min_score =', max_score, min_score)


    # max_score = 2.2

    # scores = score_map / max_score
    scores = score_map / max_score
    scores_min = np.minimum(scores, 1)

    # scores, Array of float32, (83,224,224) 


    


        
    # fig=plt.figure(figsize=(20, 15))
    # columns = 4; rows = 4

    # for j in range(0,16):
    #     colored_image = cm.jet(scores_min[j+70])
    #     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
    #     fig.add_subplot(rows, columns, j+1)
    #     padim_array = np.array(padim_image)
    #     plt.imshow(padim_array)
    #     plt.axis(False)
    # plt.show()

    # fig.savefig('../demo_image_3.jpg')

    
    fail = []
    threshhold = 100/max_score
    # print(scores_min.shape[0])
    for j in range(0, scores_min.shape[0]):
        max_value = np.max(scores_min[j])
        if max_value > threshhold:
            has_positive = True
        else:
            has_positive = False
            
        if has_positive == True:
            fail.append(j)


    loaded_data = all_location
    for i in range(0, len(fail)):
        all_fails.append([loaded_data['location'].iloc[fail[i]].large_x, loaded_data['location'].iloc[fail[i]].large_y])
        # print('region_' + str(loop) + '_img_loc: ', loaded_data['location'].iloc[i].large_x)

# 添加紅色方塊
    image = cv2.imread(input_target)
    for index in range(0, len(all_fails)):
        large_x = all_fails[index][0]
        large_y = all_fails[index][1]
        large_x = int(large_x)
        large_y = int(large_y)
        # 紅色
        image[large_y-16:large_y+16,large_x-16:large_x+16,2] = 255
    
    cv2.imwrite('../images/partno1/pred/'+direction+'.png', image)
    
    return len(fail)


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
