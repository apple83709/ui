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



    
# =============================================================================
# 
# load the source and destination images and positions
# 
# =============================================================================
def main(direction, prod_name):
# direction = 'N'
    data_combined_dir = '../../result/04_CombinedData/'+prod_name+'/region_'+direction+'_'
    for loop in range(1, 7):
        FileLoc = '../../result/05_RegionImage/'+prod_name
        try:
            os.mkdir(FileLoc)
        except OSError:
            print ("Creation of the directory %s failed" % FileLoc)
        else:
            print ("Successfully created the directory %s " % FileLoc)
            
        FileLoc = '../../result/05_RegionImage/'+prod_name+'/Region_'+direction+'_'+str(loop)+'_src'
        try:
            os.mkdir(FileLoc)
        except OSError:
            print ("Creation of the directory %s failed" % FileLoc)
        else:
            print ("Successfully created the directory %s " % FileLoc)
    
    
        FileLoc = '../../result/05_RegionImage/'+prod_name+'/Full_'+direction+'_'+str(loop)+'_src'
        try:
            os.mkdir(FileLoc)
        except OSError:
            print ("Creation of the directory %s failed" % FileLoc)
        else:
            print ("Successfully created the directory %s " % FileLoc)
    
        
        infile = data_combined_dir + str(loop) + '_img_loc.pkl'
        print(infile)
        
        
        # To load the data back
        with open(infile, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Access the data
        image_src = loaded_data['image1']
        image_dst = loaded_data['image2']
        df_location = loaded_data['location']
        Homography = loaded_data['matrix1']
        
        plt.imshow(image_src,cmap='gray')
        plt.show()
        
        plt.imshow(image_dst,cmap='gray')
        plt.show()
        
        
        # =============================================================================
        # 
        # create regional images for image_src
        # 
        # =============================================================================
        
        
        df1 = df_location.copy()
        np_src = np.zeros([len(df1),32,32]).astype(np.uint8)
        
        
        image_src_color = color_image = cv2.cvtColor(image_src, cv2.COLOR_GRAY2BGR)
        image_region = image_src_color.copy()  
        
        img = image_src
        out_region_dir = '../../result/05_RegionImage/'+prod_name+'/Region_'+direction+'_'+str(loop)+'_src/src-'
        out_full_dir = '../../result/05_RegionImage/'+prod_name+'/Full_'+direction+'_'+str(loop)+'_src/src-'
        
        out_NpFile =  '../../result/05_RegionImage/'+prod_name+'/np-src-'+direction+'-'+str(loop)+'.npy'
        
        ra = 26
        rb = 16
        
        print('\n  there are ',len(df1), 'islands\n')
        
        for i in range (0,len(df1)):
        # for i in range (8,9):
        
            r2 = df1['src_x'].loc[i]
            r1 = df1['src_y'].loc[i]
            # print('=============== i=', i,r2,r1 )
            
            if(r2>=ra and r1>=ra):
                box_image = img[r1-ra:r1+ra,r2-ra:r2+ra]
            elif(r2<ra):
                box_image = img[r1-ra:r1+ra,r2-r2:r2+r2]
            elif(r1<ra):
                box_image = img[r1-r1:r1+r1,r2-ra:r2+ra]
            # loc_image[r1-ra:r1+ra,r2-ra:r2+ra,0] = 255
        
            # plt.imshow(box_image, cmap='gray')
            # plt.show()
            
            
            # find the largest area in the box
            # Ensure it's binary
            _, binary_image = cv2.threshold(box_image, 128, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         
            min_contour_area = 30  # Minimum contour area to consider a contour as an island
            max_contour_area = 500
            
            islands = [cnt for cnt in contours if ((cv2.contourArea(cnt) > min_contour_area) \
                                                    and (cv2.contourArea(cnt) < max_contour_area) ) ]
                
            # print(len(contours), len(islands))
        
            # Draw the islands on the original image
            if box_image is not None:
                img_with_islands = cv2.cvtColor(box_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawing
                cv2.drawContours(img_with_islands, islands, -1, (255, 0, 0), 1)  # Draw islands in green color
        
            # plt.imshow(img_with_islands, cmap='gray')
            # plt.show()
          
                
            if (len(islands) > 1):
                # Filter contours based on area to find isolated islands
        
                a = 1
                df_9 = pd.DataFrame(columns=['Count', 'x', 'y', 'dist'])
        
                for k in range (0,len(islands)):
                    
                    # Initialize lists to hold the x and y coordinates
                    x_points = []
                    y_points = []
        
                    for j in range(0,len(islands[k])):
                        x = islands[k][j][0][0]
                        y = islands[k][j][0][1]
                        # print(i, j, x,y)
                        x_points.append(x)
                        y_points.append(y)
                    
                    
                    x_mean = int(np.mean(x_points))
                    y_mean = int(np.mean(y_points))
                    dist = abs(x_mean-ra) + abs(y_mean-ra)
                   
        
                    # Adding rows using 'loc'
                    df_9.loc[k] = [k, x_mean, y_mean, dist]
                    
                min_index = df_9['dist'].idxmin()
                # print('i, min_index = ', i, min_index)
                del df_9
                
            if (len(islands) == 1):
                min_index = 0
        
            if (len(islands) >0):            
                mask = np.zeros_like(box_image)
        
                # Draw the contours (islands) on the mask with white color (255) and filled (-1)
                cv2.drawContours(mask, [islands[min_index]], -1, (1), -1)    
        
                # plt.imshow(mask, cmap='gray')
                # plt.show()
              
                # Calculate moments for the largest contour
                M = cv2.moments(mask)
                
                # Calculate centroid
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0  # set to some default value if no mass found
                
                # print('cx, cy=', cx, cy)
                
                # Find 4 corners of the image and mask
                
                r3 = int(r1+cy-ra+0.5)
                r4 = int(r2+cx-ra+0.5)
                
                mask_size = mask.sum()
                
                filename = out_full_dir + str(i).zfill(4) + '-' + str(mask_size) + '.png'
                
                # print(i,r1,r2, filename)
                      
                image_region[r3-rb:r3+rb,r4-rb:r4+rb,0] = 255
                cv2.imwrite(filename, image_region)
                # plt.imshow(image_region, cmap='gray')
                # plt.show()
                     
                try:
               
                    image_input = img[r3-rb:r3+rb,r4-rb:r4+rb].copy()
                    image_mask  = mask[cy-rb:cy+rb,cx-rb:cx+rb].copy()
                    image_output = cv2.multiply(image_input, image_mask)
                    # plt.imshow(image_output, cmap='gray')
                    # plt.show()
        
                    np_src[i,:,:] = image_output[:,:]
                    
                    filename = out_region_dir + str(i).zfill(4) + '-' + str(mask_size) + '.png'
                    cv2.imwrite(filename, image_output)
                    del image_output
            
                except:
                    
                    print('------------- error in -------> ',i,r1,r2, filename)
        
        # =============================================================================
        # 
        #     for empty space, no light throught th hole, this part can be deleted
        # 
        # =============================================================================
        
            if (len(islands) ==0):            
                mask = np.zeros_like(box_image)
        
                
                filename = out_full_dir + str(i).zfill(4) + '-0.png'
                
                # print(i,r1,r2, filename)
                      
                image_region[r1-rb:r1+rb,r2-rb:r2+rb,2] = 255
                cv2.imwrite(filename, image_region)
                # plt.imshow(image_region, cmap='gray')
                # plt.show()
                     
                try:
               
                    image_output = img[r1-rb:r1+rb,r2-rb:r2+rb]
                    plt.imshow(image_output, cmap='gray')
                    plt.show()
            
                    np_src[i,:,:] = image_output[:,:]
                    filename = out_region_dir + str(i).zfill(4) + '-0.png'
                    cv2.imwrite(filename, image_output)
                    del image_output
            
                except:
                    
                    print('------------- island ==0  -------> ',i,r1,r2, filename)
        
        
        
        
        
        np.save(out_NpFile, np_src)
        
        # Load the array back from the file
        # loaded_array = np.load(out_NpFile)
        
        loaded_data['location'] = df1
        with open(infile, 'rb') as f:
            loaded_data = pickle.load(f)
        
        
        
        
