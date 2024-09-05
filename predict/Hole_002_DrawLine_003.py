# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 08:39:43 2024

draw lines on an image

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# mouse callback function
def draw_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if param['drawing'] == False:
            param['ix'], param['iy'] = x, y
            param['drawing'] = True
            # Plot a green point at the position of the first click
            cv2.circle(param['img'], (x, y), 3, (0, 255, 0), -1)
            cv2.circle(param['black_img'], (x, y), 3, (0, 255, 0), -1)
        else:
            cv2.line(param['img'], (param['ix'], param['iy']), (x, y), (0, 255, 0), 3)
            cv2.line(param['black_img'], (param['ix'], param['iy']), (x, y), (0, 255, 0), 3)
            param['coords'].append((param['line_number'], param['ix'], param['iy'], x, y))
            param['ix'], param['iy'] = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        param['drawing'] = False
        param['line_number'] += 1
        
        
# Load the first image
input_file = '../image_demo/2024-08-21_S1.BMP'

original_image = cv2.imread(input_file)

# Calculate the new dimensions
new_width = int(original_image.shape[1] * 1/4)
new_height = int(original_image.shape[0] * 1/4)

# Resize the image to two thirds of the original size
img = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
black_img = np.zeros_like(img)

param = {'drawing': False, 'ix': -1, 'iy': -1, 'img': img, 'black_img': black_img, 'coords': [], 'line_number': 0}

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line, param)

while(1):
    cv2.imshow('image', param['img'])
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break
    if param['line_number'] > 10:  # Stop after drawing 8 lines
        break

cv2.destroyAllWindows()


FileLoc = '../result/01_DrawLine'
try:
    os.mkdir(FileLoc)
except OSError:
    print ("Creation of the directory %s failed" % FileLoc)
else:
    print ("Successfully created the directory %s " % FileLoc)
    
            
# Save the displayed images
cv2.imwrite('../result/01_DrawLine/drawn_image.jpg', param['img'])
cv2.imwrite('../result/01_DrawLine/black_image_with_lines.jpg', param['black_img'])


plt.imshow(param['img'])
plt.show()

plt.imshow( param['black_img'])
plt.show()





