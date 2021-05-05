# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:38:45 2021

@author: Angga
"""

import os
import matplotlib.pyplot as plt
import  pandas as pd
from collections import namedtuple, OrderedDict
from splitting import split
from augmentation_small import copyOriginal
import shutil
import random
import numpy as np
import cv2 

def selectImage(df, target_dir, destination_dir, num_samples):
    # print(df)
    # print(target_dir)
    print(destination_dir)
    # print(num_samples)
    
    filename= []
    classname = []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    temp = []
    pick = []
    
    grouped = split(df, 'classId')
    
    
    
    # print(pick)
    for x in range(len(grouped)):    
    # for x in range(1):
        data = grouped[x][1] 
        data_list = np.array(data)
        num_data = data_list.shape[0]
        
        pick = random.sample(range(0, num_data), num_samples)
        
        # print(pick)
        
        for y in range(len(pick)):
            # print(data_list[y])
            filename.append(data_list[y][0])
            classname.append(data_list[y][1])
            width.append(int(data_list[y][2]))
            height.append(int(data_list[y][3]))
            xmin.append(int(data_list[y][4]))
            ymin.append(int(data_list[y][5]))
            xmax.append(int(data_list[y][6]))
            ymax.append(int(data_list[y][7]))
            classid.append(data_list[y][8])
            
            
            image_path = os.path.join(target_dir, data_list[y][0])
            # print(image_path)
            
            cv_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            x_min = data_list[y][4]
            y_min = data_list[y][5]
            x_max = data_list[y][6]
            y_max = data_list[y][7]
            new_image = cv_image
            new_image= cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max),(0,255,0), 1)
            # crop_image = new_image[y_min-100:y_max+100, x_min-100:x_max+100]
            # cv_resize = cv2.resize(crop_image, (300,300))
            cv2.imwrite(destination_dir+'/'+data_list[y][0], new_image)
            # cv2.imwrite(destination_dir+'/'+data_list[y][0], crop_image)
            pick.clear()
    
                  
    
    # cv2.imshow('',new_image)
    # cv2.waitKey()
            # copyOriginal(data_list[y][0], target_dir, destination_dir)
        
        
        
  
    
    
    csv = {'image_name' : filename, 'class' : classname, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classid': classid}
    df = pd.DataFrame(csv)
    dir_csv = destination_dir+'/'+'Test.csv'
    df.to_csv(dir_csv, index = False)
    
    return print("Sukses")
    
    
if __name__ == "__main__":
    
    # file = 'Images\\GTSDB\\RGB_0x0_NoCrop_Single\\GTSDB.csv'
    # file = 'Images\\GTSDB\\RGB_0x0_NoCrop_Some\\GTSDB.csv'
    file = 'Test.csv'
    folder = 'Test'
    dest = 'test'
    num_samples = 50

    root_dir = 'Images\\GTSDB\\RGB_300x300_NoCrop_Some'
    target_dir = os.path.join(os.getcwd(), root_dir, folder)
    destination_dir = os.path.join(os.getcwd(), dest)
    csv_dir = os.path.join(os.getcwd(),root_dir, folder, file)
    # csv = pd.read_csv(csv_dir)
    csv = pd.read_csv(csv_dir, sep=';')
    # print(csv)
    # annotation = csv['filename'].count()
    # classes = csv['classId'].unique()
    # image = len(csv['filename'].unique())
    # # Prohibitory =len(csv[csv['classId']==1])
    # # Mandatory = len(csv[csv['classId']==2])
    # # Danger =len(csv[csv['classId']==3])
    # # Other =len(csv[csv['classId']==4])
    # # print(Prohibitory, Mandatory, Danger, Other)
    
    # csv['classId'].hist(bins=40)
    selectImage(csv, target_dir, destination_dir, num_samples)