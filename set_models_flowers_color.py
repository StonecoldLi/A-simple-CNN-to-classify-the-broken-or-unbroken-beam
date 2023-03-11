# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:11:04 2022

@author: lijin
"""

import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time

from tensorflow.keras.callbacks import TensorBoard
import random

import numpy as np
import os
import PIL
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers,models

import pathlib

X = pickle.load(open("X.pickle","rb"))
#X = pickle.load(pickle_in)
y = pickle.load(open("y.pickle","rb"))

X = np.array(X) #features
y = np.array(y) #labels

X = X/255.0 #将数据归纳为(0,1)范围内

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

CATEGORIES = ['broken','unbroken']
NAME = 'color'
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#input_layer = tf.keras.layers.Input([80,80,3])
#model = mo
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1),
                              padding='same', activation=tf.nn.relu, input_shape=(128,128,3)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(keras.layers.Conv2D(filters=96, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(keras.layers.Conv2D(filters=96, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=2, activation=tf.nn.softmax))
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=8,callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(x=X_test, y=y_test)
print("Test Accuracy %.2f" % test_acc)

# 开始预测

correct_count = 0
predictions = model.predict(X_test)
TP = [0]*5
FP = [0]*5
TN = [0]*5
FN = [0]*5
y_pred = []
y_test1 = []
for i in range(len(X_test)):
    target = np.argmax(predictions[i])
    y_pred.append(int(target))
    label = y_test[i] #注意！！！！！！！！！！！！！
    y_test1.append(int(label))
    if target == label:
        correct_count += 1
        for i in range(5):
            if i == label:
                TP[label] += 1
            else:
                TN[label] += 1
    else:
        for i in range(5):
            if i == label:
                FP[label] += 1
            else:
                FN[label] += 1

def compute_matrics(TP, FP, TN, FN):
    accuracy = (TP + TN) /(TP + FP + TN + FN)
    precision = TP /(TP + FP)
    recall = TP /(TP + FN)
    f1_score = 2*(precision*recall)/(precision+recall)
    print("[accuracy] " + str(accuracy))
    print("[precision] " + str(precision))
    print("[recall] " + str(recall))
    print("[f1_score] " + str(f1_score))

for i in range(2):
    print("------------" + str(i) + "------------")
    compute_matrics(TP[i], FP[i], TN[i], FN[i])
print("correct prediction of total : %.2f" % (correct_count / len(X_test)))

model.save('test_beam.h5')

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test1, y_pred, np.array(['1','2']))