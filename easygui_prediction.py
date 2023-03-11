# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:24:56 2022

@author: lijin
"""

import cv2
import tensorflow as tf
import easygui as gg
import numpy as np

CATEGORIES = ['broken','unbroken']


#rescale and graysize the image
def prepare_filepath(filepath):
    IMG_SIZE = 128 #要与前面定义的一致！！！！！
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

'''
def read_random_img(path):
    """
    图片解析展示
    :param null;
    :return null;
    """
    target_shape = (128,128)
    #target_shape = (255,255)
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, target_shape, 3)
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='binary')
    plt.show()
'''


tt = "Beam Prediction Experiment ----- Made by: 高压电组"
msg = gg.msgbox(msg="Hello, Welcome to this Beam Prediction Experiment",
                title=tt,
                ok_button="Let's go!")

path = gg.enterbox(msg='Please Enter the path of the picture', 
                   title=tt, 
                   default='', 
                   strip=True, 
                   image=None,
                   )

gg.msgbox(msg="Your Picture is:",
          title=tt,
          image = path,
          ok_button="Ready to Test!!!")


model = tf.keras.models.load_model("test_beam.h5")

prediction = model.predict([prepare_filepath(path)])
#print(CATEGORIES[int(prediction[0][2])])
#print(CATEGORIES[int(np.argmax(prediction[0]))])

gg.msgbox(msg="For your picture, our model's prediction of each types are:\n Broken:{},\n Unbroken:{}".format(prediction[0][0],prediction[0][1]), 
              title=tt, ok_button='OK', image=None, root=None)

gg.msgbox(msg="In conclusion, our model's prediction of your picture is {}!".format(CATEGORIES[int(np.argmax(prediction[0]))]), 
              title=tt, ok_button='OK', image=path, root=None)
