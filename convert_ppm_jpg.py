# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:53:36 2020

@author: angga
"""


import os
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from collections import namedtuple, OrderedDict
import shutil
from create_pbtxt import labelText,labelOneClass
 
def convert_train_data(file_dir, mode, norm, fix_width, fix_height, cropmode):
 
    mode_name = getMode(mode)
    
    norm_name = getNorm(norm)
    
    crop_name = getCrop(cropmode)
   
    folder = 'Train'
    
    sub_folder = createSubFolder(mode_name, norm_name, fix_width, fix_height, crop_name, folder)
#    print(sub_folder)
    root_dir = os.path.join(os.getcwd(),'Images/GTSRB')
    # root_dir = r'C:/RTTSR/Images/GTSRB'
    root_dir += sub_folder
    
    csv_name = 'Train.csv' 

    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)    
         # This is the root directory where the picture is converted to jpg and saved as. You need to create it yourself before running the program
 
    directories = [file for file in os.listdir(file_dir)  if os.path.isdir(os.path.join(file_dir, file))]
    # print(directories)
    
    j=0
    for files in directories:
#        path = os.path.join(root_dir,files)
#        if not os.path.exists(path):
#            os.makedirs(path)
        # print( path)
                
#        folder
        data_dir = os.path.join(file_dir, files)
#        print(data_dir)
                 # print(data_dir), part of the output is as follows:
       
#        ppm
        file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir)  if f.endswith(".ppm")]
                 # Each element in file_name is the absolute address of the file suffixed with .ppm
#        print(file_names)
        
#        
        csv = [os.path.join(data_dir, f) for f in os.listdir(data_dir)  if f.endswith(".csv")]
#        print(csv)
        
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                csv_dir = os.path.join(data_dir, f)
                                 # Get the absolute address of the annotation file
#                                  print(csv_dir), part of the display is as follows:
            
#                print(csv_dir)
                
        csv_data = pd.read_csv(csv_dir)
#        print(csv_data)
                 # csv_data is a data structure in the form of DataFrama
 
        csv_data_array = np.array(csv_data)
#        print(csv_data_array.shape[0])
#        csv_data.to_csv(root_dir+'/'+ 'Train.csv', index = False)
                 # print(csv_data_array), part of the display is as follows:
     
        
        
        for i in range(csv_data_array.shape[0]):
            csv_data_list = np.array(csv_data)[i,:].tolist()[0].split(";")
#                          print(csv_data_list), part of the display is as follows:
#            print(csv_data)
 
            sample_dir = os.path.join(data_dir, csv_data_list[0])
                         # Get the absolute address of each picture in the data_dir directory
                         # print(sample_dir), part of the display is as follows:
            
            #   Information from csv
            name = csv_data_list[0]
            w = int(csv_data_list[1])
            h = int(csv_data_list[2])
            x_min = int(csv_data_list[3])
            y_min = int(csv_data_list[4])
            x_max = int(csv_data_list[5])
            y_max = int(csv_data_list[6])
            class_id = int(csv_data_list[7]) 
        
            # read file
            cv_image = readfile(sample_dir, mode, norm)    
            
            #crop roi
            cv_crop = crop(cv_image, cropmode, x_min, x_max, y_min, y_max)
            
            cv_resize, new_width, new_height, new_xmin, new_xmax, new_ymin, new_ymax = resize(cv_crop, cropmode, w, h, fix_width, fix_height, x_min, x_max, y_min, y_max)
            
            print(str(j).zfill(5))
            filename.append(name.split(".")[0]+ "_" + str(j).zfill(5) +".jpg")
            width.append(new_width)
            height.append(new_height)
            xmin.append(new_xmin)
            xmax.append(new_xmax)
            ymin.append(new_ymin)
            ymax.append(new_ymax)
            classid.append(int(class_id)+1)
            
            
#             cv_norm = normalization(cv_resize)
    #   save image
            new_dir = os.path.join(root_dir, csv_data_list[0].split(".")[0]+ "_" + str(j).zfill(5) + ".jpg")
#            cv2.imwrite(new_dir, cv_image) 
            cv2.imwrite(new_dir, cv_resize) 
    
        #   Show image    
#            test = cv2.imread(new_dir)
#            temp = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
#        #    cv2.imshow('test', temp)
#            
#        
#        #    img = cv2.rectangle(temp.copy(),(xmin, ymin), (xmax,ymax),(255,255,255), 1)
#            img = cv2.rectangle(temp.copy(),(new_xmin, new_ymin), (new_xmax,new_ymax),(0,255,0), 2)
#            plt.imshow(img)
            j+=1
#        cv2.imshow('test', img)   
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    print(filename, width, height, xmin, ymin, xmax, ymax, classid )
    csv = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
    
    df = pd.DataFrame(csv)
    df = df.sample(frac=1)
    df.to_csv(root_dir+ '/' + csv_name, index = False, sep=';')
    print("Convert Successful")

def convert_test_data(file_dir, mode, norm, fix_width, fix_height, cropmode):
    
    mode_name = getMode(mode)
    
    norm_name = getNorm(norm)
    
    crop_name = getCrop(cropmode)
   
    folder = 'Test'
    
    sub_folder = createSubFolder(mode_name, norm_name, fix_width, fix_height, crop_name, folder)
#    print(sub_folder)
    root_dir = os.path.join(os.getcwd(),'Images/GTSRB')
    # root_dir = r'C:/RTTSR/Images/GTSRB'
    root_dir += sub_folder
    
    csv_name = 'Test.csv' 
    
    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    
    
    if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
    for f in os.listdir(file_dir):
        if f.endswith(".csv"):
            csv_dir = os.path.join(file_dir, f)
            
#    print(csv_dir)
            
    csv_data = pd.read_csv(csv_dir)
    csv_data_array = np.array(csv_data)
#    csv_data.to_csv(root_dir+'/'+ 'Test.csv', index = False)
 
    for i in range(csv_data_array.shape[0]):
        csv_data_list = np.array(csv_data)[i, :].tolist()[0].split(";")
        sample_dir = os.path.join(file_dir, csv_data_list[0])
            
    #   Information from csv
        name = csv_data_list[0]
        w = int(csv_data_list[1])
        h = int(csv_data_list[2])
        x_min = int(csv_data_list[3])
        y_min = int(csv_data_list[4])
        x_max = int(csv_data_list[5])
        y_max = int(csv_data_list[6])
        class_id = int(csv_data_list[7]) 
        
        
        # read file
        cv_image = readfile(sample_dir, mode, readfile)
        
        # cv_equalhist = equalHist(cv_image)
        #crop roi
        cv_crop = crop(cv_image, cropmode, x_min, x_max, y_min, y_max)
        
        
        #resize image
        cv_resize, new_width, new_height, new_xmin, new_xmax, new_ymin, new_ymax = resize(cv_crop, cropmode, w, h, fix_width, fix_height, x_min, x_max, y_min, y_max)
     
        
        filename.append(name.split(".")[0]+".jpg")
        width.append(new_width)
        height.append(new_height)
        xmin.append(new_xmin)
        xmax.append(new_xmax)
        ymin.append(new_ymin)
        ymax.append(new_ymax)
        classid.append(int(class_id)+1)
        
        
#        cv_norm = normalization(cv_resize)
    #   save image
        new_dir = os.path.join(root_dir, csv_data_list[0].split(".")[0] + ".jpg")
    #    cv2.imwrite(new_dir, cv_image) 
        cv2.imwrite(new_dir, cv_resize) 
        
        # image = cv2.rectangle(cv_resize,(new_xmin, new_ymin), (new_xmax,new_ymax),(0,255,0), 1)
        # cv2.imwrite(new_dir, image)
        
    #   Show image    
        # test = cv2.imread(new_dir)
        # temp = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    #    cv2.imshow('test', temp)
#    
    # img = cv2.rectangle(temp.copy(),(x_min, y_min), (x_max,y_max),(255,255,255), 1)
        # img = cv2.rectangle(temp,(new_xmin, new_ymin), (new_xmax,new_ymax),(0,255,0), 2)
        # cv2.imwrite(new_dir, img)
#    #
        # plt.imshow(img)
#    cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
 
    print(filename, width, height, xmin, ymin, xmax, ymax, classid )
    csv = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
    
    df = pd.DataFrame(csv)
    df = df.sample(frac=1)
    df.to_csv(root_dir+ '/' + csv_name, index = False, sep=';')
    print("Convert Successful")

def convert_data_GTSDB(file_dir, mode, norm, fix_width, fix_height, cropmode):
    
    mode_name = getMode(mode)
    
    norm_name = getNorm(norm)
    
    crop_name = getCrop(cropmode)
    
    new_folder = ['Train' , 'Test', 'Validation']
   
    folder = ''
    
    sub_folder = createSubFolder(mode_name, norm_name, fix_width, fix_height, crop_name, folder)
#    print(sub_folder)
    root_dir = os.path.join(os.getcwd(),'Images/GTSDB')
    # root_dir = r'C:/RTTSR/Images/GTSDB'
    root_dir += sub_folder
    
    csv_train = 'Train.csv'
    csv_test = 'Test.csv'
    
    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    temp = []
    
    if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
    for f in os.listdir(file_dir):
        if f.endswith(".txt") and f == 'gt.txt':
            csv_dir = os.path.join(file_dir, f)
                
            
            csv_data = pd.read_csv(csv_dir,header=None)
    
            csv_data_array = np.array(csv_data)
            # print(csv_data_array, len(csv_data_array))
            # csv_data.to_csv(root_dir+'/'+ 'gtsdb.csv', index = False)
            
            for i in range(csv_data_array.shape[0]):
                csv_data_list = np.array(csv_data)[i, :].tolist()[0].split(";")
                sample_dir = os.path.join(file_dir, csv_data_list[0])
                # print(csv_data_list)
                    
                
                # read file
                cv_image = readfile(sample_dir, mode, readfile)
                # #   Information from csv
                name = csv_data_list[0]
                w = int(cv_image.shape[1])
                h = int(cv_image.shape[0])
                x_min = int(csv_data_list[1])
                y_min = int(csv_data_list[2])
                x_max = int(csv_data_list[3])
                y_max = int(csv_data_list[4])
                class_id = int(csv_data_list[5]) 
                # print(name, w, h, x_min, y_min, x_max, y_max, class_id)
        
        
                # cv_equalhist = equalHist(cv_image)
                #crop roi
                cv_crop = crop(cv_image, cropmode, x_min, x_max, y_min, y_max)
                
                
                # #resize image
                cv_resize, new_width, new_height, new_xmin, new_xmax, new_ymin, new_ymax = resize(cv_crop, cropmode, w, h, fix_width, fix_height, x_min, x_max, y_min, y_max)
             
                
                filename.append(name.split(".")[0]+".jpg")
                width.append(new_width)
                height.append(new_height)
                xmin.append(new_xmin)
                xmax.append(new_xmax)
                ymin.append(new_ymin)
                ymax.append(new_ymax)
                classid.append(int(class_id)+1)
                temp.append(labelOneClass(int(class_id)+1))
                
                # cv_norm = normalization(cv_resize)
            #   save image
                new_dir = os.path.join(root_dir, csv_data_list[0].split(".")[0] + ".jpg")
            #    cv2.imwrite(new_dir, cv_image) 
                cv2.imwrite(new_dir, cv_resize) 
                
                # image = cv2.rectangle(cv_resize,(new_xmin, new_ymin), (new_xmax,new_ymax), (0,255,0), 1)
                # cv2.imwrite(new_dir, image)
                
            #   Show image    
                # test = cv2.imread(new_dir)
                # temp = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
            #    cv2.imshow('test', temp)
        #    
            # img = cv2.rectangle(temp.copy(),(x_min, y_min), (x_max,y_max),(255,255,255), 1)
                # img = cv2.rectangle(temp,(new_xmin, new_ymin), (new_xmax,new_ymax),(0,255,0), 2)
                # cv2.imwrite(new_dir, img)
        #    #
                # plt.imshow(img)
            # cv2.imshow('test', cv_image)
            # cv2.waitKey(0)
        #    cv2.imshow('test', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #    cv2.waitKey(0)
        #    cv2.destroyAllWindows()
         
            # print(filename, width, height, xmin, ymin, xmax, ymax, classid )
            csv = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
            csv2 = {'image_name' : filename, 'class' : temp, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax}
            
            df = pd.DataFrame(csv)
            df2 = pd.DataFrame(csv2)
            # df = df.sample(frac=1)
            dir_csv = root_dir+'GTSDB.csv'
            dir_csv2 = root_dir+'GTSDB2.csv'
            df.to_csv(dir_csv, index = False)
            df2.to_csv(dir_csv2, index = False)
            # print("Convert Successful")
            
        
            split_data(df, root_dir, new_folder[0], 0, 591)
            split_data(df, root_dir, new_folder[1], 591, 741)
            # split_data(df, root_dir, new_folder[2], 800, 900)
            print("Convert Successful")
            
            
            
           
           
def split_data(df, root_dir, type_data, a, b):
    oneClass = 'True'
    filename= []
    width = []
    height = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classid = []
    grouped = split(df, 'filename')
    
    folder = os.path.join(root_dir, type_data)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for a in range(a,b):
        
        
        data =df[df['filename'] == grouped[a][0]]
        data_list = np.array(data)
        
        for c in range(data_list.shape[0]):
            new_data_list = np.array(data_list).tolist()
            filename.append(new_data_list[c][0])
            width.append(new_data_list[c][1])
            height.append(new_data_list[c][2])
            xmin.append(new_data_list[c][3])
            ymin.append(new_data_list[c][4])
            xmax.append(new_data_list[c][5])
            ymax.append(new_data_list[c][6])
            if oneClass =='True':
                classid.append(1)
            else:
                classid.append(new_data_list[c][7])
        
        shutil.copyfile(root_dir+grouped[a][0], folder+'/'+grouped[a][0] )
        
    
    csv_temp = {'filename' : filename, 'width' :  width, 'height' : height, 'xmin' : xmin, 'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax, 'classId' : classid}
    df_temp = pd.DataFrame(csv_temp)
  
  
    dir_csv = folder+'/'+type_data+'.csv'
    # print(df_temp)
    df_temp.to_csv(dir_csv, index = False, sep=';')
    

def split(df, group):

    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def createSubFolder(mode_name, norm_name, fix_width, fix_height, crop_name, folder):
    
    sub_folder = '/'
    sub_folder += mode_name

    if norm_name == "":
        sub_folder += '_'
    else:
        sub_folder += '_'
        sub_folder += norm_name
        sub_folder += '_'
    
    sub_folder += str(fix_width)
    sub_folder += 'x'
    sub_folder += str(fix_height)
    sub_folder += '_'
    sub_folder += crop_name
    # sub_folder += 'dummy/'
    sub_folder += '/'
    sub_folder += folder
    
    return sub_folder
    
def getCrop(crop):
    
    if crop==1:
        return "NoCrop"
    if crop==2:
        return "Crop"
    else:
        return "NoCrop"
        
def getMode(mode):
    
    if mode == 1:
        return "RGB"
    if mode == 2:
        return "Grayscale"
    if mode == 3:
        return "HSV"
    else:
        return "RGB"
    
def getNorm(norm):
    
    if norm == 1:
        return ""
    if norm == 2:
        return "MinMax"
    if norm == 3:
        return "L1"
    if norm == 4 :
        return "L2"
    else:
        return ""
    
def readfile(dir, mode, norm):
    
    cv_image = cv2.imread(dir, cv2.COLOR_BGR2RGB)
   
    cv_mode = color_mode(cv_image, mode)
    
    cv_norm = norm_mode(cv_mode, norm)
    
    return cv_norm

def color_mode(cv_image, mode):
    
    if mode == 1:
        cv_mode = cv_image
    if mode == 2:
        cv_mode = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    if mode == 3:
        cv_mode = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)   
    else:
       cv_mode = cv_image
    
    return cv_mode

def norm_mode(cv_image, norm):
     
    if norm == 1:
        cv_norm = cv_image
    if norm == 2:
        cv_norm = norm_MinMax(cv_image) 
    if norm == 3:
        cv_norm = norm_L1(cv_image)
    if norm == 4:
        cv_norm = norm_L2(cv_image) 
    else:
        cv_norm = cv_image
    
    return cv_norm
 
def norm_MinMax(cv_image):
    
    cv_norm = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv_norm

def norm_L1(cv_image):
    
    cv_norm = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_L1)
    
    return cv_norm
    
def norm_L2(cv_image):
    
    cv_norm = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_L2)
    
    return cv_norm

def equalHist(cv_image):
    
    R, G, B = cv2.split(cv_image)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    
    cv_equalhist = cv2.merge((output1_R, output1_G, output1_B))
    
    return cv_equalhist
    
def crop(cv_image, crop, xmin, xmax, ymin, ymax):
    
    if crop == 1:
        cv_crop = cv_image
    if crop == 2:    
        cv_crop = cv_image[ymin:ymax, xmin:xmax]
    else:
        cv_crop = cv_image
    return cv_crop
    
def resize(cv_image, crop, width, height, new_width, new_height, xmin, xmax, ymin, ymax) : 
    
    if new_width == 0 or new_height == 0:
        new_xmin = xmin
        new_xmax = xmax
        new_ymin = ymin
        new_ymax = ymax
        cv_resize = cv_image
    else:
        x=0
        
    #   get height
    #    y = height
        y=0
        # ratio = float(new_width/max(width, height))
        # w = int(ratio * width)
        # h = int(ratio * height)
        if crop==1:
            distx = abs(xmax-xmin)
            disty = abs(ymax-ymin)
            centx = float(distx/2)
            centy = float(disty/2)
            # scale_x = new_width/ width
            # scale_y = new_height / height
            # scale = max(scale_x, scale_y)
            # print(scale_x, scale_y)
            x_min = (xmin/width)
            x_max = (xmax/width)
            y_min = (ymin/height)
            y_max = (ymax/height)
            # print(x_min, x_max, y_min, y_max)
            new_xmin = int(x_min*new_width)
            new_xmax = int(x_max*new_width)
            new_ymin = int(y_min * new_height)
            new_ymax = int(y_max * new_height)
            # new_xmin = int(np.ceil(x_min*new_width))
            # new_xmax = int(np.ceil(x_max*new_width))
            # new_ymin = int(np.ceil(y_min * new_height))
            # new_ymax = int(np.ceil(y_max * new_height))
            # print(new_xmin, new_xmax, new_ymin, new_ymax)
            
     #   get scale
        if crop==2:
            
        
    #   scale bounding box
    #    new_xmin = int(np.round(xmin*x_scale))
    #    new_xmax = int(np.round(xmax*x_scale))
    #    new_ymin = int(np.round(ymin*y_scale))
    #    new_ymax = int(np.round(ymax*y_scale))
        
            new_xmin = x
            new_xmax = x+new_width
            new_ymin = y
            new_ymax = y+new_height
     
    #   resize image
    
        cv_resize = cv2.resize(cv_image, (new_width,new_height))
 
#    print(cv_resize.shape, cv_resize.shape[1], cv_resize.shape[0], new_xmin,new_xmax,new_ymin, new_ymax)
    return cv_resize, cv_resize.shape[1], cv_resize.shape[0], new_xmin, new_xmax,new_ymin, new_ymax
    
    
if __name__ == "__main__":
    # train_data_dir = r'C:/RTTSR/dataset/GTSRB/Final_Training/Images'
    # test_data_dir =  r'C:/RTTSR/dataset/GTSRB/Final_Test/Images'
    # gtsdb_data_dir  = r'C:/RTTSR/dataset/GTSDB'
    gtsdb_data_dir = os.path.join(os.getcwd(),'dataset/GTSDB')
    
    
    
    #mode
    #1 = RGB
    #2 = Grayscale
    #3 = HSV
    #Else = RGB    
    
    #normalization
    #1= Normal
    #2= MinMax
    #3 = L1
    #4 = L2
    #Else = normal
    
    #crop
    #1 = No Crop
    #2 = Crop
    #Else = No Crop
     
    # (dir, mode, normalization, width, height, crop)
    # convert_data_GTSDB(gtsdb_data_dir, 1, 1, 0, 0, 1)
    # convert_data_GTSDB(gtsdb_data_dir, 1, 1, 300, 300, 1)
    convert_data_GTSDB(gtsdb_data_dir, 1, 1, 300, 300, 2)
    # convert_data_GTSDB(gtsdb_data_dir, 1, 1, 1360, 800, 1)
    
  
    
    # #    RGB Normal 300x300 No Crop
    # convert_train_data(train_data_dir, 1, 1, 300, 300,1)
    # convert_test_data(test_data_dir, 1, 1, 300, 300, 1)
    
    # # #    RGB Normal 300x300 Crop
    # convert_train_data(train_data_dir, 1, 1, 300, 300,2)
    # convert_test_data(test_data_dir, 1, 1, 300, 300, 2)

    # # #    RGB Normal MinMax 300x300 No Crop
    # convert_train_data(train_data_dir, 1, 2, 300, 300,1)
    # convert_test_data(test_data_dir, 1, 2, 300, 300, 1)
    
    # # #    RGB Normal MinMax 300x300 Crop
    # convert_train_data(train_data_dir, 1, 2, 300, 300,2)
    # convert_test_data(test_data_dir, 1, 2, 300, 300, 2)
    
    # convert_train_data(train_data_dir, 1, 1, 0, 0, 1)
    # convert_test_data(test_data_dir, 1, 1, 0, 0, 1)
    
    # convert_train_data(train_data_dir, 1, 1, 0, 0, 2)
    # convert_test_data(test_data_dir, 1, 1, 0, 0, 2)
    