# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:09:49 2022

@author: lijin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "D:/datafile_concrete/file_pic" #文件所在位置
CATEGORIES = ['broken','unbroken'] #输出目标有哪些

'''
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #给出具体的图片路径
    for img in os.listdir(path): #读取目标文件夹中的文件（图片）
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #选择读取类型为灰度图读取
        plt.imshow(img_array, cmap='gray')
        plt.show() #原像素展示
        print(img_array)
        print(img_array.shape)
        break
    break

IMG_SIZE = 128 #尝试设置图片的大小尺寸

new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')#128*128像素展示
plt.show()
'''

IMG_SIZE = 128 #规定图片尺寸
training_data = [] #设置training_data list，用于存储每幅所读入图片的图片信息和标签信息

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #文件路径
        class_num = CATEGORIES.index(category) #将每类预测目标用index编码的形式表示，共5种
        print(class_num)
        #0-->daisy, 1-->dandelion,......,4-->tulips
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                #img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

#shuffle the data

import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
'''
import matplotlib.pyplot as plt
plt.imshow(sample[0], cmap='binary')
plt.show()
'''
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

#save the data

import pickle

pickle_out = open("X.pickle",'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle",'wb')
pickle.dump(y, pickle_out)
pickle_out.close()