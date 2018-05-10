# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:20:42 2017

@author: sudramak
Thanks to 
https://github.com/tkwoo/segmentation-visualization-training/
"""

from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Input,Conv2DTranspose,concatenate,Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import random
import os
import cv2
import numpy as np
from keras.preprocessing import image

image._count_valid_files_in_directory('C:\\Users\\sudramak\\SuperDS projects\\Kaggle - Caravana\train\TM1','gif','train_masks')

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def pixelwise_l2_loss(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.square(y_true_f - y_pred_f))

def pixelwise_binary_ce(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.binary_crossentropy(y_pred, y_true))



image_size = 256
row_img = image_size
col_img = image_size
lr = 0.001
img = Input((row_img,col_img,1))

conv1 = Conv2D(32,(3,3),activation=None,padding='same')(img)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Conv2D(32,(3,3),activation=None,padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = Conv2D(64,(3,3),activation=None,padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Conv2D(64,(3,3),activation=None,padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = Conv2D(128,(3,3),activation=None,padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = Conv2D(128,(3,3),activation=None,padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

conv4 = Conv2D(256,(3,3),activation=None,padding='same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
conv4 = Conv2D(256,(3,3),activation=None,padding='same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

conv5 = Conv2D(512,(3,3),activation=None,padding='same')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)
conv5 = Conv2D(512,(3,3),activation=None,padding='same')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)

up6 = Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(conv5)
up6 = concatenate([up6,conv4], axis=3)
conv6 = Conv2D(256,(3,3),padding='same')(up6)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)
conv6 = Conv2D(256,(3,3),padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)

up7 = concatenate([Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(conv6),(conv3)],axis=3)
conv7 = Conv2D(128,(3,3),padding='same')(up7)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)
conv7 = Conv2D(128,(3,3),padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)

up8 = concatenate([Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(conv7),(conv2)],axis=3)
conv8 = Conv2D(64,(3,3),padding='same')(up8)
conv8 = BatchNormalization()(conv8)
conv8 = Activation('relu')(conv8)
conv8 = Conv2D(64,(3,3),padding='same')(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Activation('relu')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(up9)
conv9 = BatchNormalization()(conv9)
conv9 = Activation('relu')(conv9)
conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Activation('relu')(conv9)

conv10 = Conv2D(1,(1,1),padding='same',activation='sigmoid')(conv9)

model = Model(inputs=img, outputs=conv10)
model.compile(optimizer=Adam(lr=lr),loss=pixelwise_binary_ce,metrics=[dice_coef])

train = ImageDataGenerator(rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                           width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                           height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                           horizontal_flip=False,  # randomly flip images
                           vertical_flip=False) 

mask = ImageDataGenerator(rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                           width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                           height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                           horizontal_flip=False,  # randomly flip imagestr
                           vertical_flip=False) 

os.chdir('C:\\Users\\sudramak\\SuperDS projects\\Kaggle - Caravana')

batch_size = 5
seed = random.randrange(1, 1000)
train_img = train.flow_from_directory('train/TT1',class_mode=None, target_size = (256,256), seed=seed, batch_size=batch_size, color_mode='grayscale')
mask_img = mask.flow_from_directory('train/TM1',class_mode=None, seed=seed, target_size=(256,256), batch_size=batch_size, color_mode='grayscale')


def train_generator(train_img,mask_img):
        while True:
            yield(next(train_img), next(mask_img))

model.fit_generator(train_generator(train_img,mask_img),steps_per_epoch=1,epochs=500)

import matplotlib.image as img
import matplotlib.pyplot as pyplt

#test_img = img.imread('train/0ce66b539f52_01.jpg')
test_img = cv2.imread('train/0cdf5b5d0ce1_01.jpg',0)
%matplotlib inline
pyplt.show(test_img)

cv2.imshow('test',test_img)

input_test = cv2.resize(test_img,(256,256))
input_test1 = cv2.resize(test_img,(256,256))
input_test = input_test.reshape((1,256,256,1))

result = model.predict(input_test)

imgMask = (result[0]*255).astype(np.uint8)
imgShow = cv2.cvtColor(input_test1, cv2.COLOR_GRAY2BGR)
_, imgMask = cv2.threshold(imgMask, int(255*0.9), 255, cv2.THRESH_BINARY)
imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)

imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.9, 0.0)

#cv2.imwrite('train/Output2.png',imgShow)
cv2.imwrite('train/Mask.png',imgMask)
#cv2.imshow('image',imgMask)

