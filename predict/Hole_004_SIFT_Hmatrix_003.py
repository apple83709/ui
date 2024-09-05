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
import time


    
    
# =============================================================================
# 
# read original and mask image
# 
# =============================================================================     

def main(input_file_source, input_file_target, direction, prod_name):
    # =============================================================================
    # 
    # create 2 directories
    # 
    # =============================================================================

    FileLoc = '../result/03_SourceTarget/'+prod_name

    try:
        os.mkdir(FileLoc)
    except OSError:
        a=1
        # print ("Creation of the directory %s failed" % FileLoc)
    # else:
        # print ("Successfully created the directory %s " % FileLoc)
        
        
    FileLoc = '../result/04_CombinedData/'+prod_name

    try:
        os.mkdir(FileLoc)
    except OSError:
        a=1
        # print ("Creation of the directory %s failed" % FileLoc)
    # else:
        # print ("Successfully created the directory %s " % FileLoc)
# input_file_source =   '../image_demo/2024-08-21_S1.BMP'
# input_file_target =   '../image_demo/2024-08-21_S2.BMP'

    image_source  = cv2.imread(input_file_source)
    image_target = cv2.imread(input_file_target)
    min_contour_area = 40  # Minimum contour area to consider a contour as an island
    max_contour_area = 180  
    out_dir = '../result/03_SourceTarget/'+prod_name+'/region_'
    data_combined_dir = '../result/04_CombinedData/'+prod_name+'/region_'+direction+'_'
    
    # plt.imshow(image_source)
    # plt.show()
    outfile = out_dir + '0_original_source.jpg'
    # print('\n',outfile)
    # cv2.imwrite(outfile, image_source)
    # plt.imshow(image_target)
    # plt.show()
    
    outfile = out_dir + '0_original_target.jpg'
    # # print(outfile)
    # cv2.imwrite(outfile, image_target)
    df_combined = pd.DataFrame()
    for loop in range (1,7):
    
        input_file_mask =      '../result/02_Region/'+prod_name+'/Region_'+direction+'_' + str(loop) + '_mask.png'
        index = 0
               
        # print('\n')
        # print(index,input_file_mask)
    
        image_mask = cv2.imread(input_file_mask)
        
        new_width = int(image_source.shape[1])
        new_height = int(image_source.shape[0])
        
        # Resize the image to two thirds of the original size
        image_mask_a = cv2.resize(image_mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
    
        # plt.imshow(image_mask_a)
        # plt.show()
        index = index + 1
        outfile = out_dir +  str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, image_mask_a)
    
    
        # =============================================================================
        # 
        # do image multiplication
        # 
        # =============================================================================
        
        # Perform element-wise multiplication
        image_mask_b = image_mask_a / 255
        image_mask_b = image_mask_b.astype(np.uint8)
        
        
        image_result_source  = cv2.multiply(image_source, image_mask_b)
        image_result_target = cv2.multiply(image_target, image_mask_b)
        
        
        # plt.imshow(image_result_source)
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, image_result_source)
        
        # plt.imshow(image_result_target)
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, image_result_target)
        
    
    # =============================================================================
    #     
    #  find 4 corners of the mask 
    #  
    # =============================================================================
     
    
    
        # Find contours
        contours, _ = cv2.findContours(image_mask_b[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assuming the mask is a single, connected component
        # Find the rotated rectangle that encloses the contour
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        
        # # print the coordinates
        # # print('Coordinates of the corners:', box)
    
        cv2.drawContours(image_result_source, [box], 0, (0, 255, 255), 4)
        # plt.imshow(image_result_source)
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, image_result_source) 
    
        
        cv2.drawContours(image_result_target, [box], 0, (0, 255, 255), 4)
        # plt.imshow(image_result_target)
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, image_result_target) 
    
    
    # =============================================================================
    # 
    # generate images of the mask for source and target
    # 
    # =============================================================================
    
        y1 = box[:,1].min()
        y2 = box[:,1].max()
        
        x1 = box[:,0].min()
        x2 = box[:,0].max()
    
        
        gray1 = image_result_source[:,:,0].copy()
        gray2 = gray1[y1:y2,x1:x2]
        
        # plt.imshow(gray1,cmap='gray')
        # plt.show()
    
        # plt.imshow(gray2,cmap='gray')
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, gray2) 
    
        gray5 = image_result_target[:,:,0].copy()
        gray6 = gray5[y1:y2,x1:x2]
        
        # plt.imshow(gray6,cmap='gray')
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, gray6)     
    
    
    # =============================================================================
    # 
    # find circles on gray-2, which is from image_result_source
    # 
    # =============================================================================
    
    
        _, thresholded  = cv2.threshold(gray2, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # plt.imshow(thresholded ,cmap='gray')
        
        # Show the plot
        # plt.show()
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, thresholded)     
        
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        islands = [cnt for cnt in contours if ((cv2.contourArea(cnt) > min_contour_area) and (cv2.contourArea(cnt) < max_contour_area)) ]
        
        # Draw the islands on the original image
        img_with_islands = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawing
        cv2.drawContours(img_with_islands, islands, -1, (225, 0, 255), 6)  # Draw islands in green color
        # Display the contour
        # plt.imshow(img_with_islands)
        # plt.show()
        
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, img_with_islands)     
        
        
        # # print(f"Found {len(islands)} islands in the image.")
        
        # Create an empty DataFrame with column names
        df_2 = pd.DataFrame(columns=['Count', 'src_x', 'src_y'])
        
        index_1 = 0
        for cnt in islands:
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # To avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
            # # print(index,cX,cY)
            index_1 = index_1 + 1
            df_2.loc[index_1] = [index_1, cX, cY]
            
        # =============================================================================
        # 
        # get the H matrix
        # 
        # =============================================================================
        
        # =============================================================================
        # 
        # Initialize SIFT detector, get aligned image
        # 
        # =============================================================================
        
        sift = cv2.SIFT_create()
        
        # Detect keypoints and descriptors
        s_time = time.time()
        keypoints1, descriptors1 = sift.detectAndCompute(gray2, None)
        e_time = time.time()
        print('*********', e_time-s_time)
        
        keypoints2, descriptors2 = sift.detectAndCompute(gray6, None)
        
        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Compute homography
        if len(good_matches) > 4:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp image
        height, width = gray2.shape[:2]
        aligned_image = cv2.warpPerspective(gray2, homography, (width, height))
        
        
        # plt.imshow(aligned_image,cmap='gray')
        # plt.show()
        
        
        
        # =============================================================================
        # 
        # transform some coordinates from source to target
        # 
        # =============================================================================
        
        
        # Assuming points_src is a list of points in the source image
        points_src = df_2[['src_x','src_y']].values.astype(float)
        
        # Add an extra dimension for homogenous coordinates
        points_src_hom = np.ones((3, len(points_src)))
        points_src_hom[:2, :] = np.transpose(points_src)
        
        # Apply the homography matrix
        points_dst_hom = np.dot(homography, points_src_hom)
        
        # Convert back to non-homogenous coordinates
        points_dst = np.zeros((len(points_src), 2))
        points_dst[:, 0] = points_dst_hom[0, :] / points_dst_hom[2, :]
        points_dst[:, 1] = points_dst_hom[1, :] / points_dst_hom[2, :]
        
        
        # =============================================================================
        # 
        # plot the locations
        # 
        # =============================================================================
        
        
        dst_x = np.zeros(len(points_dst))
        dst_y = np.zeros(len(points_dst))
        
        for i in range (0,len(points_dst)):
            r1 = points_dst[i]
            dst_x[i] = r1[0]
            dst_y[i] = r1[1]
            # # print(i,loc_x[i],loc_y[i])
            
        
        df_2['dst_x'] = dst_x.astype(np.uint)
        df_2['dst_y'] = dst_y.astype(np.uint)
        df_2 = df_2.reset_index()
        
        
        # fig = plt.figure(figsize=(6,12))
        
        # plt.tight_layout()
        # plt.show()
        
        # ax1 = plt.subplot2grid((1,1), (0,0), colspan=1)
        
        
        # df_2.plot.scatter(x='src_x', y='src_y', color = 'g', lw = 4, ax=ax1,grid=True) 
        # df_2.plot.scatter(x='dst_x', y='dst_y', color = 'r', lw = 4, ax=ax1,grid=True) 
        # plt.show()
        
        # =============================================================================
        # 
        # plot points_dst to the target image gray-6
        # 
        # =============================================================================
        
               
        src_image = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        
        
        for i in range (0,len(points_src)):
        # for i in range (0,1):
            r1 = int(points_src[i][0])
            r2 = int(points_src[i][1])
            
            src_image[r2-16:r2+16,r1-16:r1+16,0] = 255
               
            
        # Display the mask
        # plt.imshow(src_image)
        # plt.show()
            
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, src_image)    
        
        dst_image = cv2.cvtColor(gray6, cv2.COLOR_GRAY2BGR)
        
        for i in range (0,len(points_dst)):
        # for i in range (0,1):
            r1 = int(points_dst[i][0])
            r2 = int(points_dst[i][1])
            
            dst_image[r2-16:r2+16,r1-16:r1+16,2] = 255
               
            
        # Display the mask
        # plt.imshow(dst_image)
        # plt.show()
        
            
        index = index + 1
        outfile = out_dir + str(loop) + '_' + str(index).zfill(2) + '.jpg'
        # print(index, outfile)
        # cv2.imwrite(outfile, dst_image)    
        
    
    # =============================================================================
    # 
    # save gray2, gray6, H matrix, points_src, points_dst to files
    # 
    # =============================================================================
    
        index = index + 1
        outfile = data_combined_dir + str(loop) + '_img_loc.pkl'
        # print(index, outfile)

            
        df_location = df_2
        df_location = df_location.drop(columns=['index'])
        df_location['large_x'] = df_location['dst_x'] + x1
        df_location['large_y'] = df_location['dst_y'] + y1
        
        # Store the data in a dictionary
        data_combined = {
            'image1': gray2,
            'image2': gray6,
            'location': df_location,
            'matrix1': homography
        }
        
        df_combined = pd.concat([df_combined, df_location], axis=0, ignore_index=True)
        # print('len(df_combined):=========', len(df_combined))
        # Save the data to a file using pickle
        with open(outfile, 'wb') as f:
            pickle.dump(data_combined, f)
        
        # To load the data back
        with open(outfile, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Access the data
        loaded_image1 = loaded_data['image1']
        loaded_image2 = loaded_data['image2']
        loaded_location = loaded_data['location']
        # print(loaded_location.shape)
        loaded_matrix = loaded_data['matrix1']
    data_combined2 = {
        'location': df_combined
    }
    return data_combined2
    # with open('../result/04_CombinedData/'+prod_name+'/region_'+direction+'_img_loc.pkl', 'wb') as f:
    #     pickle.dump(data_combined2, f)
        
    # print('len(df_combined):=========', len(df_combined))

# main('../../trainning/image_demo/2024-08-22_N1.BMP', '../images/partno1/sample/N.BMP', 'N')
    
    
