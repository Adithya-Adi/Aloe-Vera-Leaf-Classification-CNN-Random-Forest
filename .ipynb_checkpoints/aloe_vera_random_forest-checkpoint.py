# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 09:04:44 2023

@author: adith
"""

import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
import cv2
import random
from sklearn.ensemble import RandomForestClassifier
os.environ['KMP_DUPLICATE_LIB_OK']='True'

plt.figure(figsize=(20,20))
test_folder=r'C:\Users\adith\Documents\Aloe_Vara_leaf_classification\aloe_vera\0'
for i in range(5):
    file = random.choice(os.listdir(test_folder))
    image_path= os.path.join(test_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)

IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'C:\Users\adith\Documents\Aloe_Vara_leaf_classification\aloe_vera'

classes = ["Infected","Healthy"]
def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name
X_train, y_train =create_dataset(r'C:\Users\adith\Documents\Aloe_Vara_leaf_classification\aloe_vera')
X_test, y_test =create_dataset(r'C:\Users\adith\Documents\Aloe_Vara_leaf_classification\aloe_vera_test')

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = y_train.reshape(-1,)
y_train = y_train.astype(np.int64)


print(X_train)