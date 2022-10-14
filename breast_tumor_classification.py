

! pip install kaggle

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download  forderation/breakhis-400x
! unzip breakhis-400x.zip

train_dir = '/content/BreaKHis 400X/test'
test_dir = '/content/BreaKHis 400X/train'

"""## Libraries"""

import numpy as np
import tensorflow as tf
import keras

print ("TensorFlow version: " + tf.__version__)
print ("Keras version: " + keras.__version__)

from keras import models
from keras import layers

"""## Drawing plots functions"""

import matplotlib.pyplot as plt  # library for plotting math functions: https://matplotlib.org/stable/index.html
    
def PlotAccuracyComparison(acc, val_acc, lab = '*'):
    plt.clf()   # clear figure
    #plt.rcParams['figure.figsize'] = (25.0, 5.0) # set default size of plots
    plt.figure(figsize=(25,5))
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training accuracy for ' + lab)
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy for ' + lab)
    plt.ylim(0,1)
    plt.title('Comparison of Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
def PlotLossComparison(loss, val_loss, lab = '*'):
    plt.clf()   # clear figure
    #plt.rcParams['figure.figsize'] = (25.0, 5.0) # set default size of plots
    plt.figure(figsize=(25,5))
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r', label='Training loss for ' + lab)
    plt.plot(epochs, val_loss, 'b', label='Validation loss for ' + lab)
    # plt.ylim(0,1)
    plt.title('Comparison of Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

"""## Model 5"""

from tensorflow import keras

train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='binary',
    batch_size=16,
    image_size=(64, 64))
test_ds = keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels='inferred',
    label_mode='binary',
    batch_size=16,
    image_size=(64, 64))

from keras import regularizers
from tensorflow import keras
from keras import layers

img_rows, img_cols = 64, 64

model5 = models.Sequential()
model5.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(32, (3, 3), activation='relu'))

model5.add(layers.Flatten())
model5.add(layers.Dropout(rate = 0.2))

model5.add(layers.Dense(activation = 'relu', units = 64, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(activation = 'relu', units = 32, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(16, activation='relu'))
model5.add(layers.Dense(1, activation='sigmoid'))

model5.summary()

model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history10 = model5.fit_generator(train_ds,
                        epochs = 6,
                        steps_per_epoch=4,
                        validation_data=test_ds,
                        validation_steps = 3)

acc10 = history10.history['acc']
val_acc10 = history10.history['val_acc']
loss10 = history10.history['loss']
val_loss10 = history10.history['val_loss']

PlotAccuracyComparison(acc10, val_acc10, lab = 'Model 5 - No ImageDataGenerator, bigger learning rate')

PlotLossComparison(loss10, val_loss10, lab = 'Model 5 - No ImageDataGenerator, bigger learning rate')

"""## Another way to prepare the data"""

import os
import gc
import cv2
import json
import math
import scipy
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers
from keras import layers
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras import models
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
from keras import backend as K
from functools import partial
from collections import Counter

def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
           
            img = cv2.resize(img, (RESIZE,RESIZE))
           
            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader('/content/BreaKHis 400X/train/benign',224))
malign_train = np.array(Dataset_loader('/content/BreaKHis 400X/train/malignant',224))

benign_test = np.array(Dataset_loader('/content/BreaKHis 400X/test/benign',224))
malign_test = np.array(Dataset_loader('/content/BreaKHis 400X/test/malignant',224))

# Create labels
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, 
    test_size=0.2
)

plt.imshow(X_test[6])

"""## Model 5 - another aproach"""

from keras import regularizers
from tensorflow import keras
from keras import layers

img_rows, img_cols = 224, 224

model5 = models.Sequential()
model5.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(32, (3, 3), activation='relu'))

model5.add(layers.Flatten())
model5.add(layers.Dropout(rate = 0.2))

model5.add(layers.Dense(activation = 'relu', units = 64, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(activation = 'relu', units = 32, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(16, activation='relu'))
model5.add(layers.Dense(2, activation='sigmoid'))

model5.summary()

model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history = model5.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

PlotAccuracyComparison(acc, val_acc, lab = 'Model 5')

PlotLossComparison(loss, val_loss, lab = 'Model 5 ')

"""## Model 6"""

from keras import regularizers
from tensorflow import keras
from keras import layers

img_rows, img_cols = 224, 224

model5 = models.Sequential()
model5.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model5.add(layers.MaxPooling2D((2, 2)))
model5.add(layers.Conv2D(32, (3, 3), activation='relu'))

model5.add(layers.Flatten())
model5.add(layers.Dropout(rate = 0.5))

model5.add(layers.Dense(activation = 'relu', units = 64, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(activation = 'relu', units = 32, kernel_regularizer=regularizers.l2(0.001)))
model5.add(layers.Dense(16, activation='relu'))
model5.add(layers.Dense(2, activation='softmax'))

model5.summary()

model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history2 = model5.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc2 = history2.history['acc']
val_acc2 = history2.history['val_acc']
loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

PlotAccuracyComparison(acc2, val_acc2, lab = 'Model 6')

PlotLossComparison(loss2, val_loss2, lab = 'Model 6 ')

results = model5.evaluate(X_test, Y_test)

predictions1 = model5.predict(X_test)

label =  {0:"benign",1:"malignant"}

from sklearn.metrics import confusion_matrix

results = confusion_matrix(Y_test.argmax(axis=1), predictions1.argmax(axis=1))

import seaborn as sn
import pandas as pd

plt.figure(figsize = (10,8))
sn.heatmap(results, annot=True, annot_kws={"size": 16}, fmt='g')

"""## Model 1"""

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Flatten the output to input data to the Dense layer
model1.add(layers.Flatten())
# Dense layers - similar to the previous model, but using the smaller number of units
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(2, activation='softmax'))

model1.summary()

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history3 = model1.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc3 = history3.history['acc']
val_acc3 = history3.history['val_acc']
loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']

PlotAccuracyComparison(acc3, val_acc3, lab = 'Model 1')

PlotLossComparison(loss3, val_loss3, lab = 'Model 1 ')

predictions2 = model1.predict(X_test)

from sklearn.metrics import confusion_matrix

results = confusion_matrix(Y_test.argmax(axis=1), predictions1.argmax(axis=1))

import seaborn as sn
import pandas as pd

plt.figure(figsize = (10,8))
sn.heatmap(results, annot=True, annot_kws={"size": 16}, fmt='g')

"""## Model 2"""

img_rows, img_cols = 224, 224

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Flatten the output to input data to the Dense layer
model2.add(layers.Flatten())
# Dense layers - similar to the previous model, but using the smaller number of units
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(2, activation='softmax'))

model2.summary()

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history4 = model2.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc4 = history4.history['acc']
val_acc4 = history4.history['val_acc']
loss4 = history4.history['loss']
val_loss4 = history4.history['val_loss']

PlotAccuracyComparison(acc4, val_acc4, lab = 'Model 2')

PlotLossComparison(loss4, val_loss4, lab = 'Model 2')

predictions = model2.predict(X_test)

from sklearn.metrics import confusion_matrix

results = confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))

import seaborn as sn
import pandas as pd

plt.figure(figsize = (10,8))
sn.heatmap(results, annot=True, annot_kws={"size": 16}, fmt='g')

"""## Model 3"""

img_rows, img_cols = 224, 224

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(32, (3, 3), activation='relu'))

model3.add(layers.Flatten())

model3.add(layers.Dense(16, activation='relu'))
model3.add(layers.Dense(2, activation='softmax'))

model3.summary()

model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history = model3.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

PlotAccuracyComparison(acc, val_acc, lab = 'Model 3')

PlotLossComparison(loss, val_loss, lab = 'Model 3')

predictions = model3.predict(X_test)

from sklearn.metrics import confusion_matrix

results = confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))

import seaborn as sn
import pandas as pd

plt.figure(figsize = (10,8))
sn.heatmap(results, annot=True, annot_kws={"size": 16}, fmt='g')

"""## Model 4"""

from keras import regularizers

img_rows, img_cols = 224, 224

model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Conv2D(32, (3, 3), activation='relu'))

model4.add(layers.Flatten())

model4.add(layers.Dense(activation = 'relu', units = 64, kernel_regularizer=regularizers.l2(0.001)))
model4.add(layers.Dense(activation = 'relu', units = 32, kernel_regularizer=regularizers.l2(0.001)))
model4.add(layers.Dense(16, activation='relu'))
model4.add(layers.Dense(2, activation='softmax'))

model4.summary()

model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

history = model4.fit(
    x= x_train, 
    y = y_train,
    batch_size = 16,
    steps_per_epoch=x_train.shape[0]/16,
    epochs=12,
    validation_data=(x_val, y_val)

)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

PlotAccuracyComparison(acc, val_acc, lab = 'Model 4')

PlotLossComparison(loss, val_loss, lab = 'Model 4')

predictions = model4.predict(X_test)

from sklearn.metrics import confusion_matrix

results = confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))

import seaborn as sn
import pandas as pd

plt.figure(figsize = (10,8))
sn.heatmap(results, annot=True, annot_kws={"size": 16}, fmt='g')