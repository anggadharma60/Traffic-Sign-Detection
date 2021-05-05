# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 22:40:35 2021

@author: Angga
"""
import pandas as pd
from collections import namedtuple, OrderedDict
import os
import numpy as np
import shutil
from sklearn.utils import shuffle
from create_pbtxt import labelText,labelOneClass, labelSomeText

def split_data(df, root_dir, type_data, train, validation, test, typeClass):
   
    train_filename= []
    train_width = []
    train_height = []
    train_xmin = []
    train_xmax = []
    train_ymin = []
    train_ymax = []
    train_classid = []
    train_temp =[]
    # 
    validation_filename= []
    validation_width = []
    validation_height = []
    validation_xmin = []
    validation_xmax = []
    validation_ymin = []
    validation_ymax = []
    validation_classid = []
    validation_temp =[]
    # 
    test_filename= []
    test_width = []
    test_height = []
    test_xmin = []
    test_xmax = []
    test_ymin = []
    test_ymax = []
    test_classid = []
    test_temp =[]
    
    grouped = split(df, 'classId')
    
    train_fol = os.path.join(root_dir, type_data[0])
    validation_fol = os.path.join(root_dir, type_data[1])
    test_fol = os.path.join(root_dir, type_data[2])
    if not os.path.exists(train_fol):
        os.makedirs(train_fol)
    if not os.path.exists(validation_fol):
        os.makedirs(validation_fol)
    if not os.path.exists(test_fol):
        os.makedirs(test_fol)
          
        
    for i in range(0,43):
        
        
        data =df[df['classId'] == grouped[i][0]]
        total = len(data)
        a = int((train * total) / 100)
        b = int((validation * total) / 100)
        c = int((test * total) / 100)
       
        num_train = a
        num_validation = b
        num_test = c
        
        data_list = np.array(data)
        new_data_list = np.array(data_list).tolist()
        
        # Train
        for x in range(0, num_train):
            train_filename.append(new_data_list[x][0])
            train_width.append(new_data_list[x][1])
            train_height.append(new_data_list[x][2])
            train_xmin.append(new_data_list[x][3])
            train_ymin.append(new_data_list[x][4])
            train_xmax.append(new_data_list[x][5])
            train_ymax.append(new_data_list[x][6])
            
            if typeClass == 2:
                train_classid.append(new_data_list[x][7])
                train_temp.append(labelText(int(new_data_list[x][7])))
            else:
                train_classid.append(1)
                train_temp.append(labelOneClass(int(new_data_list[x][7])))
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[x][0], train_fol+'/'+new_data_list[x][0] )
        
        # Validation
        for y in range(num_train, num_train+num_validation):
            validation_filename.append(new_data_list[y][0])
            validation_width.append(new_data_list[y][1])
            validation_height.append(new_data_list[y][2])
            validation_xmin.append(new_data_list[y][3])
            validation_ymin.append(new_data_list[y][4])
            validation_xmax.append(new_data_list[y][5])
            validation_ymax.append(new_data_list[y][6])
            
            if typeClass == 2:
                validation_classid.append(new_data_list[y][7])
                validation_temp.append(labelText(int(new_data_list[y][7])))
            else:
                validation_classid.append(1)
                validation_temp.append(labelOneClass(int(new_data_list[y][7])))
                
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[y][0], validation_fol+'/'+new_data_list[y][0] )
        # Test
        for z in range(num_train+num_validation, num_train+num_validation+num_test):
            test_filename.append(new_data_list[z][0])
            test_width.append(new_data_list[z][1])
            test_height.append(new_data_list[z][2])
            test_xmin.append(new_data_list[z][3])
            test_ymin.append(new_data_list[z][4])
            test_xmax.append(new_data_list[z][5])
            test_ymax.append(new_data_list[z][6])
            
            if typeClass == 2:
                validation_classid.append(new_data_list[z][7])
                validation_temp.append(labelText(int(new_data_list[z][7])))
            else:
                validation_classid.append(1)
                validation_temp.append(labelOneClass(int(new_data_list[z][7])))
                
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[z][0], test_fol+'/'+new_data_list[z][0] )
           

      
    csv_train = {'filename' : train_filename, 'class_text' : train_temp, 'width' :  train_width, 'height' : train_height, 'xmin' : train_xmin, 'ymin' : train_ymin, 'xmax' : train_xmax, 'ymax' : train_ymax, 'classId' : train_classid}
    csv_validation = {'filename' : validation_filename,  'class_text' : validation_temp, 'width' :  validation_width, 'height' : validation_height, 'xmin' : validation_xmin, 'ymin' : validation_ymin, 'xmax' : validation_xmax, 'ymax' : validation_ymax, 'classId' : validation_classid}
    csv_test = {'filename' : test_filename,  'class_text' : test_temp, 'width' :  test_width, 'height' : test_height, 'xmin' : test_xmin, 'ymin' : test_ymin, 'xmax' : test_xmax, 'ymax' : test_ymax, 'classId' : test_classid}
    
    df_train = pd.DataFrame(csv_train)
    df_validation = pd.DataFrame(csv_validation)
    df_test = pd.DataFrame(csv_test)
    
    df_train = shuffle(df_train)
    df_validation = shuffle(df_validation)
    df_test = shuffle(df_test)
    
    dir_csv_train = train_fol+'/'+type_data[0]+'.csv'
    dir_csv_validation = validation_fol+'/'+type_data[1]+'.csv'
    dir_csv_test = test_fol+'/'+type_data[2]+'.csv'
   
    df_train.to_csv(dir_csv_train, index = False, sep=';')
    df_validation.to_csv(dir_csv_validation, index = False, sep=';')
    df_test.to_csv(dir_csv_test, index = False, sep=';')
    
    print("Split data sukses")
    

def split_data_some(df, root_dir, type_data, train, validation, test, typeClass):
   
    train_filename= []
    train_width = []
    train_height = []
    train_xmin = []
    train_xmax = []
    train_ymin = []
    train_ymax = []
    train_classid = []
    train_temp =[]
    # 
    validation_filename= []
    validation_width = []
    validation_height = []
    validation_xmin = []
    validation_xmax = []
    validation_ymin = []
    validation_ymax = []
    validation_classid = []
    validation_temp =[]
    # 
    test_filename= []
    test_width = []
    test_height = []
    test_xmin = []
    test_xmax = []
    test_ymin = []
    test_ymax = []
    test_classid = []
    test_temp =[]
    
    grouped = split(df, 'classId')
    
    train_fol = os.path.join(root_dir, type_data[0])
    validation_fol = os.path.join(root_dir, type_data[1])
    test_fol = os.path.join(root_dir, type_data[2])
    if not os.path.exists(train_fol):
        os.makedirs(train_fol)
    if not os.path.exists(validation_fol):
        os.makedirs(validation_fol)
    if not os.path.exists(test_fol):
        os.makedirs(test_fol)
          
        
    for i in range(0,4):
        
        
        data =df[df['classId'] == grouped[i][0]]
        total = len(data)
        a = int((train * total) / 100)
        b = int((validation * total) / 100)
        c = int((test * total) / 100)

        num_train = 150
        num_validation = 75
        num_test = 25
        
        data_list = np.array(data)
        new_data_list = np.array(data_list).tolist()
        
        
        # Train
        for x in range(0, num_train):
            train_filename.append(new_data_list[x][0])
            train_width.append(new_data_list[x][1])
            train_height.append(new_data_list[x][2])
            train_xmin.append(new_data_list[x][3])
            train_ymin.append(new_data_list[x][4])
            train_xmax.append(new_data_list[x][5])
            train_ymax.append(new_data_list[x][6])
            
            if typeClass == 2:
                train_classid.append(new_data_list[x][7])
                train_temp.append(labelText(int(new_data_list[x][7])))
            elif typeClass == 1:
                train_classid.append(1)
                train_temp.append(labelOneClass(int(new_data_list[x][7])))
            elif typeClass == 3: 
                train_classid.append(new_data_list[x][7])
                train_temp.append(labelSomeText(int(new_data_list[x][7])))
                
                
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[x][0], train_fol+'/'+new_data_list[x][0] )
        # Validation
        for y in range(num_train, num_train+num_validation):
            validation_filename.append(new_data_list[y][0])
            validation_width.append(new_data_list[y][1])
            validation_height.append(new_data_list[y][2])
            validation_xmin.append(new_data_list[y][3])
            validation_ymin.append(new_data_list[y][4])
            validation_xmax.append(new_data_list[y][5])
            validation_ymax.append(new_data_list[y][6])
            
            if typeClass == 2:
                validation_classid.append(new_data_list[y][7])
                validation_temp.append(labelText(int(new_data_list[y][7])))
            elif typeClass == 1:
                validation_classid.append(1)
                validation_temp.append(labelOneClass(int(new_data_list[y][7])))
            elif typeClass == 3: 
                validation_classid.append(new_data_list[y][7])
                validation_temp.append(labelSomeText(int(new_data_list[y][7])))
  
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[y][0], validation_fol+'/'+new_data_list[y][0] )
        
        
        # # Test
        for z in range(num_train+num_validation, num_train+num_validation+num_test):
            test_filename.append(new_data_list[z][0])
            test_width.append(new_data_list[z][1])
            test_height.append(new_data_list[z][2])
            test_xmin.append(new_data_list[z][3])
            test_ymin.append(new_data_list[z][4])
            test_xmax.append(new_data_list[z][5])
            test_ymax.append(new_data_list[z][6])
            
            if typeClass == 2:
                test_classid.append(new_data_list[z][7])
                test_temp.append(labelText(int(new_data_list[z][7])))
            elif typeClass == 1:
                test_classid.append(1)
                test_temp.append(labelOneClass(int(new_data_list[z][7])))
            elif typeClass == 3: 
                test_classid.append(new_data_list[z][7])
                test_temp.append(labelSomeText(int(new_data_list[z][7])))
        
            shutil.copyfile(root_dir+'/augmented/'+new_data_list[z][0], test_fol+'/'+new_data_list[z][0] )
           

      
    csv_train = {'filename' : train_filename, 'class_text' : train_temp, 'width' :  train_width, 'height' : train_height, 'xmin' : train_xmin, 'ymin' : train_ymin, 'xmax' : train_xmax, 'ymax' : train_ymax, 'classId' : train_classid}
    csv_validation = {'filename' : validation_filename,  'class_text' : validation_temp, 'width' :  validation_width, 'height' : validation_height, 'xmin' : validation_xmin, 'ymin' : validation_ymin, 'xmax' : validation_xmax, 'ymax' : validation_ymax, 'classId' : validation_classid}
    csv_test = {'filename' : test_filename,  'class_text' : test_temp, 'width' :  test_width, 'height' : test_height, 'xmin' : test_xmin, 'ymin' : test_ymin, 'xmax' : test_xmax, 'ymax' : test_ymax, 'classId' : test_classid}
    
    
    df_train = pd.DataFrame(csv_train)
    # print(df_train)
    df_validation = pd.DataFrame(csv_validation)
    # print(df_validation)
    df_test = pd.DataFrame(csv_test)
    
    df_train = shuffle(df_train)
    df_validation = shuffle(df_validation)
    df_test = shuffle(df_test)
    
    dir_csv_train = train_fol+'/'+type_data[0]+'.csv'
    dir_csv_validation = validation_fol+'/'+type_data[1]+'.csv'
    dir_csv_test = test_fol+'/'+type_data[2]+'.csv'
   
    df_train.to_csv(dir_csv_train, index = False, sep=';')
    df_validation.to_csv(dir_csv_validation, index = False, sep=';')
    df_test.to_csv(dir_csv_test, index = False, sep=';')
    
    print("Split data sukses")

def split(df, group):
    data = namedtuple('data', ['classId', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]



if __name__ == "__main__":
    root_dir = []
    gtsdb_data_dir = []
    # root_dir.append('Images/GTSDB/RGB_300x300_NoCrop_Single')
    # root_dir.append('Images/GTSDB/RGB_300x300_NoCrop_Multi')
    # root_dir.append('Images/GTSDB/RGB_0x0_NoCrop_Single')
    # root_dir.append('Images/GTSDB/RGB_0x0_NoCrop_Multi')
    # root_dir.append('Images/GTSDB/RGB_300x300_Crop_Single')
    # root_dir.append('Images/GTSDB/RGB_300x300_Crop_Multi')
    root_dir.append('Images/GTSDB/RGB_0x0_NoCrop_Some')
    # root_dir.append('Images/GTSDB/RGB_300x300_NoCrop_Some')
    folder = 'augmented'
    temp = len(root_dir)
    for i in range(0, temp):
        gtsdb_data_dir.append(os.path.join(os.getcwd(),root_dir[i], folder))
    

    # new_folder = ['Train' , 'Validation', 'Test']
    # train = 70
    # validation = 20
    # test = 10
    # typeClass = [1,2,1,2]
    # temp = len(gtsdb_data_dir)
    # for x in range(0, temp):
    #     for f in os.listdir(gtsdb_data_dir[x]):
    #         if f.endswith(".csv") and f == 'GTSDB_AUG.csv':
    #             csv_dir = os.path.join(gtsdb_data_dir[x], f)
    #             df = pd.read_csv(csv_dir, sep=';')
    #             split_data(df, root_dir[x], new_folder, train, validation, test, typeClass[x])
    
    new_folder = ['Train' , 'Validation', 'Test']
    train = 70
    validation = 20
    test = 10
    typeClass = [3,3]
    temp = len(gtsdb_data_dir)
    for x in range(0, temp):
        for f in os.listdir(gtsdb_data_dir[x]):
            if f.endswith(".csv") and f == 'GTSDB_AUG.csv':
                csv_dir = os.path.join(gtsdb_data_dir[x], f)
                df = pd.read_csv(csv_dir, sep=';')
                split_data_some(df, root_dir[x], new_folder, train, validation, test, typeClass[x])
            
          
    # print(gtsdb_data_dir)
    # print(new_folder)
    