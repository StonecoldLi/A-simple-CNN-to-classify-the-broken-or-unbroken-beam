# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 17:59:11 2023

@author: lijin
"""

import PIL.Image as img
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
#im = img.open("D:/datafile_concrete/file_pic/unbroken/mv1304.jpg")

#im.rotate(30).show()

DATADIR = "D:/datafile_concrete/file_pic" #文件所在位置
CATEGORIES = ['broken','unbroken'] #输出目标有哪些

list_num = [1,2,3,4,5,6,7,8,9,0]
flag = 0

'''
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    #print(type(os.listdir(path)))
    for each in os.listdir(path):
        #flag += 1
        im = img.open(os.path.join(path,each))
        im.show()
        match = re.search(r'(\d)(\.)',each) #找寻末尾的数字编号
        for i in range(len(list_num)):
            #flag = flag + 1
            #im = img.open(os.path.join(path,each))
            #match = re.search(r'(\d)(\.)',each) #找寻末尾的数字编号
            if match.group(1) == list_num[i]: #编号相等的话
                im = im.rotate(36*i)
                im.show()
                break
            else:
                i = i + 1
    break
'''
def rotate_picture_by_number():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to 2 types of flowers
        #print(path)
        #class_num = CATEGORIES.index(category)
        #print(class_num)
        #0-->cat, 1-->dog
        for num in range (len(os.listdir(path))):
            try:
                #print(im)
                pic = img.open(path+'/'+os.listdir(path)[num])
               # print(type(re.search(r'(\d)(\.)',os.listdir(path)[num]).group(1))
                #print(type(im))
                match = re.search(r'(\d)(\.)',os.listdir(path)[num])
               # print(match.group(1))
                #pic.show()
               # print('1')
                #print(list_num)
                for each in list_num:
                    print(match.group(1),str(each))
                    if match.group(1) == str(each):
                        print("1")
                        pic = pic.rotate(36*int(each))
                        pic.save(path+'/'+os.listdir(path)[num])
                        break
                
            
            except Exception as e:
                pass
            

rotate_picture_by_number()

