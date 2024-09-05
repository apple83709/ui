# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""


import random
from random import sample
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import warnings
warnings.filterwarnings('ignore')
import time


# device setup

BATCH_SIZE = 32 # please try 8, 16, 32, 64

device = 'cpu'
# device = 'cuda'
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

  
####################################################################
#
# load the Resnet-18 model:
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


# load model

model = resnet18(pretrained=True, progress=True)
t_d = 448
t_d = 192
d = 50
model.to(device)
model.eval()
random.seed(1024)
torch.manual_seed(1024)


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


# =============================================================================
# 
# load the X_tain and X_test files
# 
# =============================================================================


test_image_in_numpy = np.random.rand(840,3,224,224)
train_image_in_numpy = np.random.rand(840,3,224,224)


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

print("\n====>   train_dataloader")

# Create the dataset
dataset = MyDataset(torch.from_numpy(train_image_in_numpy).float(), torch.from_numpy(train_target_in_numpy).float())

# Create the DataLoader
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
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

    B, C, H, W = embedding_vectors.size()
    print("embedding_vectors.size() = ", embedding_vectors.size())


    embedding_vectors = embedding_vectors.view(B, C, H * W)
    print("embedding_vectors.size() = ", embedding_vectors.size())

    
    embedding_vectors_train = embedding_vectors.clone().numpy()
    
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    
    mean_train = mean.copy()
    
    
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    
    
    train_outputs = [mean, cov]
    
    print("mean.shape, cov.shape = ", mean.shape, cov.shape)


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
test_dataloader = DataLoader(dataset_2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

gt_list = []
gt_mask_list = []
test_imgs = []

print("\n====>   test_dataloader")

index = 0        
       
# extract test set features
for (x, _) in tqdm(test_dataloader):
    
    index = index + 1
    
    with torch.no_grad():
        _ = model(x.to(device))
        
    # get intermediate layer outputs   
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())
    # initialize hook outputs
    outputs = []
        
for k, v in test_outputs.items():
    test_outputs[k] = torch.cat(v, 0)
    print(k)

end_time = time.time()
execution_time = end_time - start_time
print(f"\n*** Program execution time (first part) is: {execution_time} seconds\n")
start_time = end_time


# =============================================================================
    
# Embedding concat
embedding_vectors = test_outputs['layer1']


for layer_name in ['layer2']:
    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])


print("embedding_vectors.shape = ", embedding_vectors.shape)


# randomly select d dimension
embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
print("embedding_vectors.shape = ", embedding_vectors.shape)


# calculate distance matrix
B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
print("B, C, H, W = ", B, C, H, W)
print("embedding_vectors.shape = ", embedding_vectors.shape)

# =============================================================================
# 
# calculate mahalanobis distance
# 
# =============================================================================


print("\n====>   mahalanobis distance")

dist_list = []

# for i in range(H * W):
for i in range(2304):
    
    dist = []

    conv_inv = np.linalg.inv(train_outputs[1][:, :, i])

    sample_train = embedding_vectors_train[:,:,i]
    sample_test  = embedding_vectors[:,:,i]
    sample_diff = sample_test - sample_train   
    maha_full = sample_diff @ conv_inv @ (sample_diff.T)
    
    # print(index,sample_test.shape,sample_train.shape, sample_diff.shape, maha_left.shape)

    for k in range(0, len(train_image_in_numpy)):
        distance = maha_full[k,k]
        dist.append(distance)
 
    dist_list.append(dist)


# dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
# print("dist_list.shape",dist_list.shape)

end_time = time.time()
execution_time = end_time - start_time

print(f"\n*** Program execution time (second part) is: {execution_time} seconds\n")

