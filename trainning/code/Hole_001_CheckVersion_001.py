# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:53:07 2024

Python version: 3.9.19 (main, Mar 21 2024, 17:21:27) [MSC v.1916 64 bit (AMD64)]
cv2 version: 4.10.0
torch version: 2.2.0+cu121
torchvision version: 0.17.0+cu121

@author: USER
"""

import sys
print("Python version: {}". format(sys.version))

import cv2
print("cv2 version: {}".format(cv2.__version__))

import torch as tor
print("torch version: {}".format(tor.__version__))

import torchvision as torvis
print("torchvision version: {}".format(torvis.__version__))

