# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 22:07:05 2023

@author: adith
"""

import numpy as np
from tensorflow.keras import layers, models
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report
import os
import cv2
import random
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

classes = ["Rust","Healthy"]
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

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = y_test.reshape(-1,)
y_test = y_test.astype(np.int64)

X_train = X_train / 255.0

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test,y_test)

y_pred = cnn.predict(X_test)
print(y_pred[:5])

y_classes = [np.argmax(element) for element in y_pred]

plot_sample(X_test, y_test,41)
pred_classes = ["Rust","Healthy"]
print(classification_report(y_test, y_classes, target_names=pred_classes))

X_testt, y_testt =create_dataset(r'C:\Users\adith\Documents\Aloe_Vara_leaf_classification\atest')
X_testt = np.array(X_testt)
y_predd = cnn.predict(X_testt)
y_classes = [np.argmax(element) for element in y_predd]
print(y_classes)
