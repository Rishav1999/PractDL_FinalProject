#!/usr/bin/env python
# coding: utf-8

# ### Practical Deep Learning System Performance: Final Project
# #### Topic: Applications of Deep Learning in Healthcare: Breast Cancer Classification using Deep Learning models
# #### Rishav Agarwal (ra3141)
# #### Rachana Dereddy (rd2998)

# In[102]:


import glob 
import random
from matplotlib.image import imread
import cv2
from keras.preprocessing.image import load_img, img_to_array
from numpy import concatenate
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, MaxPool2D
from keras import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.backend import learning_phase
from tensorflow import keras
from keras.layers.normalization.batch_normalization import BatchNormalization
import warnings 
warnings.filterwarnings('ignore')


# In[57]:


get_ipython().system("unzip 'data_load/12_set1.zip' -d 'data'")
get_ipython().system("unzip 'data_load/16_set2.zip' -d 'data'")
get_ipython().system("unzip 'data_load/20_set3.zip' -d 'data'")
get_ipython().system("unzip 'data_load/20_set4.zip' -d 'data'")
get_ipython().system("unzip 'data_load/20_set5.zip' -d 'data'")
get_ipython().system("unzip 'data_load/20_set5.zip' -d 'data'")
get_ipython().system("unzip 'data_load/21_set5.zip' -d 'data'")
print("Data Extracted")
#!unzip '/content/gdrive/MyDrive/PractDL/data/archive.zip' -d '/content/gdrive/MyDrive/PractDL/data'


# In[78]:


class0 = []
class1 = []
for filename in glob.glob('data1/*/*/*.png'):
    if filename.endswith('class0.png'):
        class0.append(filename)
    else:
        class1.append(filename)


# In[79]:


print("Length of class0", len(class0))
print("Length of class1", len(class1))


# In[80]:


sampled_class0 = random.sample(class0, len(class1))
sampled_class1 = random.sample(class1, len(class1))
len(sampled_class0)


# In[81]:


class0_array = []
c= 0
count = 0
from numpy import savetxt
for image in sampled_class0:
    img = load_img(image)
    c = c+1
    #if(c==100):
    #    count = count+1
        #print(count+1)
        #savetxt('data.csv', class0_array, delimiter=',')
    #    c = 0
    arrayed_image = img_to_array(img)
    arrayed_image = arrayed_image.astype('uint8')
    arrayed_image = cv2.resize(arrayed_image, (50, 50), interpolation=cv2.INTER_LINEAR)
    class0_array.append([arrayed_image,0])
print(len(class0_array))


# In[82]:


class1_array = []
c= 0
count = 0
from numpy import savetxt
for image in sampled_class1:
    img = load_img(image)
    arrayed_image = img_to_array(img)
    arrayed_image = arrayed_image.astype('uint8')
    arrayed_image = cv2.resize(arrayed_image, (50, 50), interpolation=cv2.INTER_LINEAR)
    class1_array.append([arrayed_image,1])
print(len(class1_array))


# In[103]:


class0_array = np.array(class0_array)
class1_array = np.array(class1_array)
dataset1 = concatenate((class0_array,class1_array))
print("Shape of the dataset",dataset1.shape)


# In[84]:


X = []
y = []

for pixels,classes in dataset1:
    X.append(pixels)
    y.append(classes)

#converting into numpy array
X = np.array(X)
y = np.array(y)
print("Shape of X:", X.shape)
print("Shape of y", y.shape)


# In[85]:


#train test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)

#shapes of test and train
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Shape of X train:", X_train.shape)
print("Shape of X test:", X_test.shape)
print("Shape of Y train:", y_train.shape)
print("Shape of y test:", y_test.shape)


# In[101]:


fig = plt.figure(figsize = (8,10))

print("Class Label 0:")
plt.subplot(1,2,1)
plt.imshow(class0_array[0][0])
plt.subplot(1,2,2)
plt.imshow(class0_array[1][0])


# In[100]:



print("Class Label 1:")
plt.subplot(1,2,1)
plt.imshow(class1_array[0][0])
plt.subplot(1,2,2)
plt.imshow(class1_array[1][0])


# ### VGG16

# In[108]:


#vgg-16
vgg = Sequential()
vgg.add(Conv2D(input_shape=(50,50,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vgg.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg.add(Flatten())
vgg.add(Dense(4096,activation="relu"))
vgg.add(Dense(4096,activation="relu"))
vgg.add(Dense(2, activation="sigmoid"))

vgg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

vgg.summary()


# In[109]:


history_vgg = vgg.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs= 200,
    verbose = 1,
    batch_size=250
    )


# In[111]:


history_vgg.history['val_accuracy']


# ### AlexNet Model

# In[112]:


#alexnet
alexnet = Sequential()
alexnet.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(50,50,3)))
alexnet.add(BatchNormalization())
alexnet.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
alexnet.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
alexnet.add(BatchNormalization())
alexnet.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
alexnet.add(BatchNormalization())
alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
alexnet.add(BatchNormalization())
alexnet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
alexnet.add(BatchNormalization())
alexnet.add(MaxPool2D(pool_size=(3,3), strides=(2,2),padding='same')) 
alexnet.add(Flatten())
alexnet.add(Dense(4096, activation='relu'))
alexnet.add(Dropout(0.5))
alexnet.add(Dense(4096,activation='relu'))
alexnet.add(Dropout(0.5))
alexnet.add(Dense(2,activation='softmax'))

opt = keras.optimizers.SGD(learning_rate=0.01)
alexnet.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

alexnet.summary()


# In[113]:


history_alexnet = alexnet.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs= 200,
    verbose = 1,
    batch_size=250
    )


# In[114]:


history_alexnet.history['val_accuracy']


# ### Model 3 Created

# In[115]:


#building the baseline model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(50,50,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])


model.summary()


# In[116]:


history_base = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs= 200,
    verbose = 1,
    batch_size=250
    )


# In[122]:


history_base.history['val_accuracy']


# ### Transfer Learning Model on ResNet50

# In[117]:


res50_model = ResNet50(input_shape = (50,50,3),weights=None,classes=2)

#Learning Rate: 0.001
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 180:
        lr *= 0.001
    elif epoch > 120:
        lr *= 0.01
    elif epoch > 60:
        lr *= 0.1
    print('Learning rate: ', lr)
    return lr
lr_scheduler = LearningRateScheduler(lr_schedule)

tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False, name="SGD")
res50_model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
res50_model.layers[0].trainable = False
history_tl = res50_model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_test,y_test),shuffle=True,verbose=1, callbacks=[lr_scheduler])


# In[118]:


history_tl.history['val_accuracy']


# In[40]:


df2 = pd.read_csv("Accuracy_P100.csv")


# In[52]:


print("Accuracy for P100")
df2


# In[48]:


df.to_csv("Accuracy_V100.csv", index = False)


# In[53]:


print("Accuracy for V100")
df


# In[54]:


import matplotlib.pyplot as plt
def plot_epochs():
    val_acc = df2["Model-3"]
    val_acc1 = df2["AlexNet"]
    val_acc2 = df2["Transfer Learning (ResNet50)"]
    val_acc3 = df2["VGG16"]
    
    epochs = range(len(val_acc))
    fig,ax = plt.subplots(2,1 , figsize=(5, 10))
    ax[0].set_title('Accuracy vs Epochs P100')
    
    ax[0].plot(epochs, val_acc3, color='green', label='VGG')
    ax[0].plot(epochs, val_acc1, color='orange', label='AlexNet')
    ax[0].plot(epochs, val_acc, color='blue', label='Model3')
    ax[0].plot(epochs, val_acc2, color='red', label='TL')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    val_acc = df["Model-3"]
    val_acc1 = df["AlexNet"]
    val_acc2 = df["Transfer Learning (ResNet50)"]
    val_acc3 = df["VGG16"]
    
    ax[1].set_title('Accuracy vs Epochs V100')
    
    ax[1].plot(epochs, val_acc3, color='green', label='VGG')
    ax[1].plot(epochs, val_acc1, color='orange', label='AlexNet')
    ax[1].plot(epochs, val_acc, color='blue', label='Model3')
    ax[1].plot(epochs, val_acc2, color='red', label='TL')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()


# In[55]:


plot_epochs()

