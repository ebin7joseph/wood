import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import re
import cv2
from time import time
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

paths = []
l = []
for i in range(1,13):
    temp = [0,0,0,0,0,0,0,0,0,0,0,0]
    paths.append('/WoodDataset/Test/' + str(i) + "/")
    temp[i-1] = 1;
    l.append(temp);

uri = []
for i in paths:
    uri.append([sorted(glob.glob(os.path.join(os.getcwd() + i, '*.jpg')), key=natural_key)])

width = 200
height = 100
shape = (height,width,3)
dataset = []
label = []

for u in uri:
    i = 0 
    for path in u:
      temp = l[i];
      for sub_path in path:
        img = cv2.imread(path[0])
        img = cv2.resize(img,(width,height))
        img = img_to_array(img)
        dataset.append(img)
        label.append(temp)
      i += 1




dataset = np.array(dataset, dtype="float") / 255.0
label = np.array(label, dtype = 'int')
dataset,label = shuffle(dataset,label)

model = tf.keras.models.load_model('./savedmodel/model0005-1.0000.h5')


test_loss, test_acc, test_mae = model.evaluate(dataset, label)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print('Test mae:', test_mae)
