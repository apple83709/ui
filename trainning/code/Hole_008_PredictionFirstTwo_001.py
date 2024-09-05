# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""


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


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
device = 'cpu'
print('device =', device)


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='C:/Users/USER/Documents/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='.\mvtec_result')
    # parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    return parser.parse_args()


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

  

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


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


train_feature_filepath = './train_feature_20240811.pkl'

args = parse_args()

# load model

model = resnet18(pretrained=True, progress=True)
t_d = 448
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


out_X_train_file = '../result/06_SortedTrainTest/X_train.npy'
np_1 = np.load(out_X_train_file)
train_image_in_numpy = np_1.astype(float)/1.0


out_X_test_file = '../result/06_SortedTrainTest/X_test.npy'

# out_X_test_file = '../result/06_SortedTrainTest/X_test.npy'

np_2 = np.load(out_X_test_file)
test_image_in_numpy = np_2.astype(float)/1.0


test_image_in_numpy = np.random.rand(420,3,224,224)
train_image_in_numpy = np.random.rand(420,3,224,224)





# =============================================================================
# 
# prepare input data
# 
# =============================================================================





train_target_in_numpy = np.ones((len(train_image_in_numpy), 1), dtype=np.uint8)

test_target_in_numpy = np.ones((len(test_image_in_numpy), 1), dtype=np.uint8)


# fig=plt.figure(figsize=(20, 15))
# columns = 4; rows = 4

# for j in range(0,16):
#     torch_image = train_image_in_numpy[j,:,:,:]
#     image_for_matplotlib = torch_image.transpose((1, 2, 0))
#     fig.add_subplot(rows, columns, j+1)
#     plt.imshow(image_for_matplotlib )
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()


fig=plt.figure(figsize=(20, 15))
columns = 3; rows = 2

for j in range(0,6):
    torch_image = train_image_in_numpy[j+7,:,:,:]
    image_for_matplotlib = torch_image.transpose((1, 2, 0))/255
    fig.add_subplot(rows, columns, j+1)
    plt.imshow(image_for_matplotlib, vmin=0, vmax=1)
    # plt.title(get_class_title(i) + str(img.size))
    plt.axis(False)
    # fig.add_subplot
plt.show()


fig=plt.figure(figsize=(20, 15))
columns = 3; rows = 2

for j in range(0,6):
    torch_image = test_image_in_numpy[j+7,:,:,:]
    image_for_matplotlib = torch_image.transpose((1, 2, 0))/255
    fig.add_subplot(rows, columns, j+1)
    plt.imshow(image_for_matplotlib, vmin=0, vmax=1)
    # plt.title(get_class_title(i) + str(img.size))
    plt.axis(False)
    # fig.add_subplot
plt.show()



# fig.savefig('../result/PADIM_demo_image_1.jpg')



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
    # embedding_vectors.size() =  torch.Size([209, 100, 56, 56])
    
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    print("embedding_vectors.size() = ", embedding_vectors.size())
    # embedding_vectors.size() =  torch.Size([209, 100, 3136])
    
    embedding_vectors_train = embedding_vectors.clone().numpy()
    
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    
    mean_train = mean.copy()
    
    
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    
    
    train_outputs = [mean, cov]
    
    print("mean.shape, cov.shape = ", mean.shape, cov.shape)
    # mean.shape, cov.shape =  (100, 3136) (100, 100, 3136)
    # train_output is a dictionary with two elements: mean and cov
    
    with open(train_feature_filepath, 'wb') as f:
        pickle.dump(train_outputs, f)
        
    # end_time = time.time()
    # execution_time = end_time - start_time

    # print(f"Trainning time is: {execution_time} seconds")

        
if (flag_train ==0):
    
    print("\n#3   train_feature_filepath =", train_feature_filepath)
    print('load train set feature from: %s' % train_feature_filepath)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)



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
    
    index = index + 1
    # print("\ntest, index, x, y = ", index)
    test_imgs.extend(x.cpu().detach().numpy())
    # gt_list.extend(y.cpu().detach().numpy())
    # gt_mask_list.extend(mask.cpu().detach().numpy())
    # model prediction
    
    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs   
    
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())
    # initialize hook outputs
    outputs = []

end_time = time.time()
execution_time = end_time - start_time
print(f"Trainning time is: {execution_time} seconds")
start_time = end_time
        
for k, v in test_outputs.items():
    test_outputs[k] = torch.cat(v, 0)
    print(k)

#     # test_outputs:
#     # layer1, Tensor, (69, 64, 56, 56)
#     # layer2, Tensor, (69, 128, 28, 28)
#     # layer3, Tensor, (69, 256, 14, 14)
    
# =============================================================================
#
#     test_outputs has 83 (images), 3 layers (64+128+256=448 features), 
#     feature images sizes are 56x56, 28x28, 14x14  
#     
#     in test_outputs:
#     
#     layer1: 69x64x56x56
#     layer2: 69x128x28x28
#     layer3: 69x256x14x14
#     
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
# select 100 out of 448, using idx
# embedding_vectors.shape =  torch.Size([83, 100, 56, 56])

# calculate distance matrix
B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
print("B, C, H, W = ", B, C, H, W)
print("embedding_vectors.shape = ", embedding_vectors.shape)

# B, C, H, W =  83 100 56 56
# embedding_vectors.shape =  (83, 100, 3136)

# train_outputs    
# 0, Array of float32, (100, 3136)
# 1, Array of float32, (100, 100, 3136)

# =============================================================================
# 
# calculate mahalanobis distance
# 
# =============================================================================

dist_list = []

for i in range(H * W):
# for i in range(0,5):    
    
    # mean_replace = train_outputs[0][:, i]
    conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
    
    index = 0
    dist = []
    for k in range(0, len(train_image_in_numpy)):
        
        sample_train = embedding_vectors_train[k,:,i]
        sample_test  = embedding_vectors[k,:,i]
        # print(index,sample_test.shape,sample_train.shape)
        index = index + 1
        distance = mahalanobis(sample_test, sample_train, conv_inv)
        dist.append(distance)

    
    dist_list.append(dist)
    # print(len(dist_list))
    # print(embedding_vectors.shape, len(dist),len(dist_list))
    # print(mean_replace.shape, conv_inv.shape, len(dist))




dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)


print("dist_list.shape",dist_list.shape)
# (83, 56, 56)



# upsample
dist_list = torch.tensor(dist_list)
# (83, 56, 56)

score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                          align_corners=False).squeeze().numpy()
# score_map, Array of float32, (83,224,224) 


# apply gaussian smoothing on the score map
for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    
# score_map[19] = 0
# Normalization
max_score = score_map[0:23].max()
min_score = score_map[0:23].min()

print('max_score, min_score =', max_score, min_score)


# max_score = 2.2

# scores = score_map / max_score
scores = score_map / max_score
scores_min = np.minimum(scores, 1)

# scores, Array of float32, (83,224,224) 


end_time = time.time()
execution_time = end_time - start_time

print(f"Program executed in: {execution_time} seconds")



# fig=plt.figure(figsize=(20, 15))
# columns = 3; rows = 2

# for j in range(0,6):
#     colored_image = cm.jet(scores_min[j+7])
#     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
#     fig.add_subplot(rows, columns, j+1)
#     padim_array = np.array(padim_image)
#     plt.imshow(padim_array)
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()

# fig=plt.figure(figsize=(20, 15))
# columns = 3; rows = 2

# for j in range(0,6):
#     colored_image = cm.jet(scores_min[j+17])
#     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
#     fig.add_subplot(rows, columns, j+1)
#     padim_array = np.array(padim_image)
#     plt.imshow(padim_array)
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()

# fig=plt.figure(figsize=(20, 15))
# columns = 3; rows = 2

# for j in range(0,6):
#     colored_image = cm.jet(scores_min[j+27])
#     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
#     fig.add_subplot(rows, columns, j+1)
#     padim_array = np.array(padim_image)
#     plt.imshow(padim_array)
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()

# fig=plt.figure(figsize=(20, 15))
# columns = 3; rows = 2

# for j in range(0,6):
#     colored_image = cm.jet(scores_min[j+37])
#     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
#     fig.add_subplot(rows, columns, j+1)
#     padim_array = np.array(padim_image)
#     plt.imshow(padim_array)
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()

# fig=plt.figure(figsize=(20, 15))
# columns = 3; rows = 2

# for j in range(0,6):
#     colored_image = cm.jet(scores_min[j+47])
#     padim_image = Image.fromarray((colored_image * 255).astype(np.uint8))
#     fig.add_subplot(rows, columns, j+1)
#     padim_array = np.array(padim_image)
#     plt.imshow(padim_array)
#     # plt.title(get_class_title(i) + str(img.size))
#     plt.axis(False)
#     # fig.add_subplot
# plt.show()


# # fig.savefig('../PADIM_demo_image_2.jpg')




