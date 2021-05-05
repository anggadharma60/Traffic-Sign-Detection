# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:03:33 2021

@author: Angga
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from collections import namedtuple, OrderedDict
import shutil
from create_pbtxt import labelText,labelOneClass
import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def split(df, group):
    data = namedtuple('data', ['classId', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def augmentation(path_image, csv_file, folder, total):
    dir_csv = os.path.join(path_image, csv_file)
    csv = pd.read_csv(dir_csv)
    new_folder = os.path.join(path_image, folder)
    
    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    temp = []
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    
    grouped = split(csv, 'classId')
    for a in range(0,43):
        data = grouped[a][1] 
        data_list = np.array(data)
        num_data = data_list.shape[0]
        
        for b in range(data_list.shape[0]):
            filename.append(data_list[b][0])
            width.append(int(data_list[b][1]))
            height.append(int(data_list[b][2]))
            xmin.append(int(data_list[b][3]))
            ymin.append(int(data_list[b][4]))
            xmax.append(int(data_list[b][5]))
            ymax.append(int(data_list[b][6]))
            classid.append(data_list[b][7])
            copyOriginal(data_list[b][0], path_image, new_folder)
        
        diff = total - num_data 
        for c in range(0, diff):
            rand_index = random.randrange(num_data)
            selected = data_list[rand_index]
            img = cv2.imread(path_image+'/'+selected[0])
            name = selected[0].split(".")[0]+"_"+str(c)+".jpg"
            w = selected[1]
            h = selected[2]
            x_min= selected[3]
            y_min = selected[4]
            x_max = selected[5]
            y_max = selected[6]
            class_id = selected[7] 
            
            
 
            img_aug, new_xmin, new_ymin, new_xmax, new_ymax = random_augmentator(img, x_min, y_min, x_max, y_max)
            # print(new_xmin, new_ymin, new_xmax, new_ymax)
            
            filename.append(name)
            width.append(int(w))
            height.append(int(h))
            xmin.append(new_xmin)
            ymin.append(new_ymin)
            xmax.append(new_xmax)
            ymax.append(new_ymax)
            classid.append(int(class_id))
            
            # cv2.rectangle(img_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
            # plt.imshow(img_aug)
            new_dir = os.path.join(new_folder,name)
            cv2.imwrite(new_dir, img_aug)
        
        # print(num_data)
        # print(diff)
        # print(len(xmin))
    
    csv_aug = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
    df_aug = pd.DataFrame(csv_aug)
    csv_aug_name = 'GTSDB_AUG.csv'
    dir_csv_aug = os.path.join(new_folder, csv_aug_name)
    df_aug.to_csv(dir_csv_aug, index = False, sep=';')
    
    # cv2.imshow('',img_aug)
    # cv2.waitKey()
    # print(filename, width, height, xmin, ymin, xmax, ymax, classid)
    # print(data_list)
    
    # print(xmin, ymin, xmax, ymax)
    # print(bbs_aug)
    # print(selected)
    # print(total)
    # print(num_data)
    # print(rand_index)
   
    print("Augmentation Success")
    return 

def augmentationSomeClass(path_image, csv_file, folder, total):
    dir_csv = os.path.join(path_image, csv_file)
    csv = pd.read_csv(dir_csv)
    new_folder = os.path.join(path_image, folder)
    
    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    temp = []
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    
    grouped = split(csv, 'classId')
    for a in range(0,4):
        data = grouped[a][1] 
        data_list = np.array(data)
        num_data = data_list.shape[0]
        
        for b in range(data_list.shape[0]):
            filename.append(data_list[b][0])
            width.append(int(data_list[b][1]))
            height.append(int(data_list[b][2]))
            xmin.append(int(data_list[b][3]))
            ymin.append(int(data_list[b][4]))
            xmax.append(int(data_list[b][5]))
            ymax.append(int(data_list[b][6]))
            classid.append(data_list[b][7])
            copyOriginal(data_list[b][0], path_image, new_folder)
        
        diff = total - num_data 
        for c in range(0, diff):
            rand_index = random.randrange(num_data)
            selected = data_list[rand_index]
            img = cv2.imread(path_image+'/'+selected[0])
            name = selected[0].split(".")[0]+"_"+str(c)+".jpg"
            w = selected[1]
            h = selected[2]
            x_min= selected[3]
            y_min = selected[4]
            x_max = selected[5]
            y_max = selected[6]
            class_id = selected[7] 
            
            
 
            img_aug, new_xmin, new_ymin, new_xmax, new_ymax = random_augmentator(img, x_min, y_min, x_max, y_max)
            # print(new_xmin, new_ymin, new_xmax, new_ymax)
            
            filename.append(name)
            width.append(int(w))
            height.append(int(h))
            xmin.append(new_xmin)
            ymin.append(new_ymin)
            xmax.append(new_xmax)
            ymax.append(new_ymax)
            classid.append(int(class_id))
            
            # cv2.rectangle(img_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
            # plt.imshow(img_aug)
            new_dir = os.path.join(new_folder,name)
            cv2.imwrite(new_dir, img_aug)
        
        # print(num_data)
        # print(diff)
        # print(len(xmin))
    
    csv_aug = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
    df_aug = pd.DataFrame(csv_aug)
    csv_aug_name = 'GTSDB_AUG.csv'
    dir_csv_aug = os.path.join(new_folder, csv_aug_name)
    df_aug.to_csv(dir_csv_aug, index = False, sep=';')
    
    # cv2.imshow('',img_aug)
    # cv2.waitKey()
    # print(filename, width, height, xmin, ymin, xmax, ymax, classid)
    # print(data_list)
    
    # print(xmin, ymin, xmax, ymax)
    # print(bbs_aug)
    # print(selected)
    # print(total)
    # print(num_data)
    # print(rand_index)
   
    print("Augmentation Success")
    return 

def copyOriginal(filename, path_image, new_folder):
    
    shutil.copy(path_image+'/'+filename, new_folder+'/'+ filename)
    # print(filename)

def aug_flip(img, x_min, y_min, x_max, y_max):
    
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Fliplr(p = 1.0) # apply horizontal flip
        ])
    
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
  
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_avgblur(img, x_min, y_min, x_max, y_max):
    
    kernel = random.randint(2,5)
    # print(kernel)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.AverageBlur(k=(kernel)) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax
    
def aug_medblur(img, x_min, y_min, x_max, y_max):
    
    
    kernel = [3, 5] 
    i = random.randint(0,1)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.MedianBlur(k=(kernel[i])) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_add(img, x_min, y_min, x_max, y_max):
    
     
    i = random.randint(-75,75)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Add(i) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_mul(img, x_min, y_min, x_max, y_max):
    
    c = [0.5, 0.75, 1.25, 1.50, 1.75, 2.00]
    i = random.randint(0,5)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Multiply(c[i]) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_scale(img, x_min, y_min, x_max, y_max):
    
    c = [0.75, 0.8, 0.85, 0.9, 0.95]
    i = random.randint(0,4)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Affine(scale=(c[i])) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_rotate(img, x_min, y_min, x_max, y_max):
    
    i = random.randint(-15,15)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.95),rotate=(i)) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_shear(img, x_min, y_min, x_max, y_max):
    
    i = random.randint(-8,8)
    # print(i)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.9),shear=(i)) # apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_translate(img, x_min, y_min, x_max, y_max):
    
    x = random.randint(-25,25)
    y = random.randint(-25,25)
    # print(x,y)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.Affine(scale=(0.9)),
        iaa.TranslateX(px=(x)),
        iaa.TranslateY(px=(y))# apply horizontal flip
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def aug_histEqu(img, x_min, y_min, x_max, y_max):
    
 
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
        ], shape=img.shape)
    
    seq = iaa.Sequential([
        iaa.HistogramEqualization()
        ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    
    new_xmin = int(bbs_aug[0][0][0])
    new_ymin = int(bbs_aug[0][0][1])
    new_xmax = int(bbs_aug[0][1][0])
    new_ymax = int(bbs_aug[0][1][1])
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def random_augmentator(img, x_min, y_min, x_max, y_max):
    while (True):
        x = random.randint(0,2)
        y = random.randint(0,2)
        z = random.randint(0,5)
        
        if (x!=0 or y!=0 or z!=0):
            break
    image_aug, new_xmin, new_ymin, new_xmax, new_ymax = pick_blur(x, img, x_min, y_min, x_max, y_max)
    image_aug, new_xmin, new_ymin, new_xmax, new_ymax = pick_contras(y, image_aug, new_xmin, new_ymin, new_xmax, new_ymax)
    image_aug, new_xmin, new_ymin, new_xmax, new_ymax = pick_operation(z, image_aug, new_xmin, new_ymin, new_xmax, new_ymax)
    
    # cv2.rectangle(image_aug,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
    # plt.imshow(image_aug)
    
    return image_aug, new_xmin, new_ymin, new_xmax, new_ymax

def pick_blur(x, img, x_min, y_min, x_max, y_max):
    if x==0:
        return img, x_min, y_min, x_max, y_max
    if x==1:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_avgblur(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==2:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_medblur(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax 

def pick_contras(x, img, x_min, y_min, x_max, y_max):
    if x==0:
        return img, x_min, y_min, x_max, y_max
    if x==1:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_add(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==2:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_mul(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    # if x==3:
    #     img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_histEqu(img, x_min, y_min, x_max, y_max)
    #     return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
        
def pick_operation(x, img, x_min, y_min, x_max, y_max):
    if x==0:
        return img, x_min, y_min, x_max, y_max
    if x==1:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_flip(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==2:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_scale(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==3:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_rotate(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==4:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_shear(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
    if x==5:
        img_aug, new_xmin, new_ymin, new_xmax, new_ymax = aug_translate(img, x_min, y_min, x_max, y_max)
        return img_aug, new_xmin, new_ymin, new_xmax, new_ymax
        
        

        
    
    
if __name__ == "__main__":
    path = []
    path.append('Images\GTSDB\RGB_0x0_NoCrop_Some') 
    # path.append('Images\GTSDB\RGB_300x300_NoCrop_Some')
    csv_file = 'GTSDB.csv'
    folder = 'augmented'
    total = 600
    # single = os.path.join(os.getcwd(), path[0])
    # multi = os.path.join(os.getcwd(), path[1])
    some1 = os.path.join(os.getcwd(), path[0])
    # some2 = os.path.join(os.getcwd(), path[1])
    augmentationSomeClass(some1, csv_file, folder, total)
    # some = os.path.join(os.getcwd(), path[0])
    # augmentationSomeClass(some2, csv_file, folder, total)
    # augmentation(multi, csv_file, folder, total)
    # img = cv2.imread(dir_img)
    # cv2.imshow('test',img)
    # cv2.waitKey()
    # plt.imshow(img)