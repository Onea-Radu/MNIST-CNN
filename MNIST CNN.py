#%%
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology,feature
(Xtrain,ytrain),(Xtest,ytest),=keras.datasets.mnist.load_data()





Xtrain=np.asarray([x>filters.threshold_otsu(x) for x in Xtrain],dtype=np.float32)

dilatated=np.asarray([morphology.binary_dilation(x) for x in Xtrain],dtype=np.float32)
eroded=np.asarray([morphology.binary_erosion(x) for x in Xtrain],dtype=np.float32)



Xtest=np.asarray([x>filters.threshold_otsu(x) for x in Xtest],dtype=np.float32)
dilatatedt=np.asarray([morphology.binary_dilation(x) for x in Xtest],dtype=np.float32)
erodedt=np.asarray([morphology.binary_erosion(x) for x in Xtest],dtype=np.float32)


if keras.backend.image_data_format() == 'channels_first':
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, 28, 28)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, 28, 28)
    dilatated = dilatated.reshape(Xtrain.shape[0], 1, 28, 28)
    dilatatedt = dilatatedt.reshape(Xtest.shape[0], 1, 28, 28)
    eroded = eroded.reshape(Xtrain.shape[0], 1, 28, 28)
    erodedt = erodedt.reshape(Xtest.shape[0], 1, 28, 28)



    input_shape = (1, 28, 28)
else:
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)
    dilatated = dilatated.reshape(Xtrain.shape[0],  28, 28,1)
    dilatatedt = dilatatedt.reshape(Xtest.shape[0], 28, 28,1)
    eroded = eroded.reshape(Xtrain.shape[0], 28, 28,1)
    erodedt = erodedt.reshape(Xtest.shape[0], 28, 28,1)
    input_shape = (28, 28, 1)

inputnormal=keras.layers.Input(input_shape)
inputeroded=keras.layers.Input(input_shape)
inputdilatated=keras.layers.Input(input_shape)                            
normal1=keras.layers.Conv2D(70,3,activation='relu')(inputnormal)
normal2=keras.layers.Conv2D(150,3,activation='relu')(normal1)
max=keras.layers.MaxPooling2D(2)(normal2)



flattened=keras.layers.Flatten()(max)
drop=keras.layers.Dropout(0.3)(flattened)
dense1=keras.layers.Dense(300,'relu')(drop)
dense2=keras.layers.Dense(300,'relu')(dense1)
drop1=keras.layers.Dropout(0.5)(dense2)
dense3=keras.layers.Dense(200,'relu')(drop1)
drop2=keras.layers.Dropout(0.3)(dense3)
dense5=keras.layers.Dense(100,'relu')(drop2)
softmax=keras.layers.Dense(10,'softmax')(dense5)
model=keras.models.Model((inputnormal,inputeroded),softmax)
model.compile(keras.optimizers.Adamax(),loss='categorical_crossentropy',metrics=['acc'])
model.fit((Xtrain,eroded),keras.utils.to_categorical(ytrain),epochs=20,verbose=2,validation_data=((Xtest,erodedt),keras.utils.to_categorical(ytest)),use_multiprocessing=True,batch_size=200)


#model.evaluate((Xtest,erodedt),keras.utils.to_categorical(ytest))








# %%
