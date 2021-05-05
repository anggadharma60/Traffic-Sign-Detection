# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:48:35 2020

@author: Angga
"""

import pandas as pd
import os

def labelOneClass(classId):
    classMember = []
    for i in range(1,44):
        classMember.append(i)
    if classId in classMember:
        return 'Traffic-Sign'
    # if classId
    
def someClass(classId):
    
    Prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16] 
    Mandatory = [33, 34, 35, 36, 37, 38, 39, 40] 
    Danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    if classId in Prohibitory:
        return 1
    elif classId in Mandatory:
        return 2
    elif classId in Danger:
        return 3
    else:
        return 4
        
def labelSomeText(classId):
    if classId == 1:
        return 'Prohibitory'
    if classId == 2:
        return 'Mandatory'
    if classId == 3:
        return 'Danger'
    if classId == 4:
        return 'Other'
    
def labelText(classId):
    if classId == 1:
        return 'speed limit 20'
    if classId == 2:
        return 'speed limit 30'
    if classId == 3:
        return 'speed limit 50'
    if classId == 4:
        return 'speed limit 60'
    if classId == 5:
        return 'speed limit 70'
    if classId == 6:
        return 'speed limit 80'
    if classId == 7:
        return 'restriction ends 80'
    if classId == 8:
        return 'speed limit 100'
    if classId == 9:
        return 'speed limit 120'
    if classId == 10:
        return 'no overtaking'
    if classId == 11:
        return 'no overtaking (trucks)'
    if classId == 12:
        return 'priority at next intersection'
    if classId == 13:
        return 'priority road'
    if classId == 14:
        return 'give way'
    if classId == 15:
        return 'stop'
    if classId == 16:
        return 'no traffic both ways'
    if classId == 17:
        return 'no trucks'
    if classId == 18:
        return 'no entry'
    if classId == 19:
        return 'danger'
    if classId == 20:
        return 'bend left'
    if classId == 21:
        return 'bend right'
    if classId == 22:
        return 'bend'
    if classId == 23:
        return 'uneven road'
    if classId == 24:
        return 'slippery road'
    if classId == 25:
        return 'road narrows'
    if classId == 26:
        return 'construction'
    if classId == 27:
        return 'traffic signal'
    if classId == 28:
        return 'pedestrian crossing'
    if classId == 29:
        return 'school crossing'
    if classId == 30:
        return 'cycles crossing'
    if classId == 31:
        return 'snow'
    if classId == 32:
        return 'animals'
    if classId == 33:
        return 'restriction ends' 
    if classId == 34:
        return 'go right'
    if classId == 35:
        return 'go left'
    if classId == 36:
        return 'go straight'
    if classId == 37:
        return 'go right or straight'
    if classId == 38:
        return 'go left or straight'
    if classId == 39:
        return 'keep right'
    if classId == 40:
        return 'keep left'
    if classId == 41:
        return 'roundabout'
    if classId == 42:
        return 'restriction ends (overtaking)'
    if classId == 43:
        return 'restriction ends (overtaking (trucks))'

def createPBTXT(dir, new_dir):
    
    output=''
    
    df = pd.read_csv(dir,sep=';')
    # print(df)
    
    classId = df.sort_values(by=['classId'], ascending=True)
    class_num = classId.classId.unique()
    # print(class_num)
    for i in class_num:
        print(i,labelText(i))
        output +="item { \n" +  \
        "  id: " + str(i) + "\n" \
        +  "  name: \'" + labelText(i) + '\'' +  "\n}\n\n"
        
    # print(output)
    myFile = open(new_dir, 'w+')
    myFile.write(output)
    
def createOneClassPBTXT(dir, new_dir):
    
    output=''
    
    df = pd.read_csv(dir,sep=';')
    # print(df)
    
    classId = df.sort_values(by=['classId'], ascending=True)
    class_num = classId.classId.unique()
    # print(class_num)
    for i in class_num:
        print(i,labelOneClass(i))
        output +="item { \n" +  \
        "  id: " + str(i) + "\n" \
        +  "  name: \'" + labelOneClass(i) + '\'' +  "\n}\n\n"
        
    # print(output)
    myFile = open(new_dir, 'w+')
    myFile.write(output)

def createSomeClassPBTXT(dir, new_dir):
    
    output=''
    
    df = pd.read_csv(dir,sep=';')
    # print(df)
    
    classId = df.sort_values(by=['classId'], ascending=True)
    # print(classId)
    class_num = classId.classId.unique()
    # print(class_num)
    for i in class_num:
        print(i,labelSomeText(i))
        output +="item { \n" +  \
        "  id: " + str(i) + "\n" \
        +  "  name: \'" + labelSomeText(i) + '\'' +  "\n}\n\n"
        
    # print(output)
    myFile = open(new_dir, 'w+')
    myFile.write(output)
    
def createSomeClassTFLite(dir, new_dir):
    
    output=''
    
    df = pd.read_csv(dir,sep=';')
    # print(df)
    
    classId = df.sort_values(by=['classId'], ascending=True)
    # print(classId)
    class_num = classId.classId.unique()
    # print(class_num)
    for i in class_num:
        # print(i,labelSomeText(i))
        output += labelSomeText(i) +  "\n"
        
    # print(output)
    myFile = open(new_dir, 'w+')
    myFile.write(output)
    
if __name__ == "__main__":
    
    # example_single = 'Images/GTSDB/RGB_300x300_NoCrop_Single/Train/Train.csv'
    # example_multi = 'Images/GTSDB/RGB_300x300_NoCrop_Multi/Train/Train.csv'
    example_some = 'Images/GTSDB/RGB_300x300_NoCrop_Some/Train/Train.csv'
    
    root_dir = os.path.join(os.getcwd() , 'data')
    # single = 'SingleClass_label.pbtxt'
    # multi = 'MultiClass_label.pbtxt'
    # some = 'SomeClass_label.pbtxt'
    TFLite = 'TFLite_label.txt'
    
    
    # new_dir_single = os.path.join(root_dir, single) 
    # new_dir_multi = os.path.join(root_dir, multi) 
    new_dir_some = os.path.join(root_dir, TFLite) 
    # print(new_dir_single)
   
    # createPBTXT(example_multi, new_dir_multi)
    # createOneClassPBTXT(example_single, new_dir_single)
    # createSomeClassPBTXT(example_some, new_dir_some)
    createSomeClassTFLite(example_some, new_dir_some)