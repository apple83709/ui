# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:03:53 2024


@author: USER
"""


import cv2
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os
import warnings 
warnings.filterwarnings('ignore')


input_file_1 = '../image_demo/2024-08-21_S1.BMP'
input_file_2 = '../result/01_DrawLine/drawn_image.jpg'
input_file_3 = '../result/01_DrawLine/black_image_with_lines.jpg'

FileLoc = '../result/02_Region'
try:
    os.mkdir(FileLoc)
except OSError:
    print ("Creation of the directory %s failed" % FileLoc)
else:
    print ("Successfully created the directory %s " % FileLoc)
    
            


image_1 = cv2.imread(input_file_1)
image_2 = cv2.imread(input_file_2)
image_3 = cv2.imread(input_file_3)

image_3a = image_3[:,:,0] 

plt.imshow(image_1)
plt.show()
plt.imshow(image_2)
plt.show()
plt.imshow(image_3a,cmap='gray')
plt.show()


# =============================================================================
# 
# find circles
# 
# =============================================================================



_, thresh  = cv2.threshold(image_3a, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)


kernel = np.ones((5,5), np.uint8)

# Apply dilation
dilated_mask = cv2.dilate(thresh, kernel, iterations = 1)


# thresh_inv = 255-thresh
plt.imshow(dilated_mask,cmap='gray')

# Show the plot
plt.show()



# Find contours
contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 100  # Minimum area to be considered a region
regions = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]



# For each region
for i, region in enumerate(regions):
    # Create a black image
    mask = np.zeros_like(image_3a)
    
    # Draw the filled contour on the black image
    cv2.drawContours(mask, [region], -1, (255), thickness=cv2.FILLED)
    
    # Save the mask image
    cv2.imwrite(f'../result/02_Region/region_{i}_mask.png', mask)
    plt.imshow(mask,cmap='gray')
    
    # Show the plot
    plt.show()
    print('region = ',i)




