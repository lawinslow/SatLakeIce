# Convolution neural network classifier for frozen/non frozen lakes
# Trained with Keras on TensorFlow backend (image dim ordering is (height, width, channels))

from __future__ import print_function

import joblib
import numpy as np
import os
from sklearn.model_selection import ShuffleSplit

from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

summer_data = joblib.load("./data/train_summer_data.joblib")
winter_data = joblib.load("./data/train_winter_data.joblib")

# assign labels
# summer = 0, winter = 1
y_summer = np.zeros((summer_data.shape[0],))
y_winter = np.ones((winter_data.shape[0],))

# concatenate data
X = np.vstack((summer_data, winter_data))
y = np.hstack((y_summer, y_winter))

# scale images to 0..1
X_max = np.max(X)

# in order to get a fair validation,
# dump the max value from the training data
joblib.dump(X_max, ".data/X_max.joblib")

# split off a dev set that we won't train on,
# as a validation set
ss = ShuffleSplit(n_splits = 1)
for train_indx, dev_indx in ss.split(X):
    X_train = X[train_indx]
    y_train = y[train_indx]
    X_dev = X[dev_indx]
    y_dev = y[dev_indx]

# here we do some tricks to get more training data,
# scaling, shearing, rotating, flipping
# (which should still look like 'lakes')
train_datagen = ImageDataGenerator(
    rescale=1./X_max,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=360)

train_datagen.fit(X_train, augment = True)

dev_datagen = ImageDataGenerator(rescale=1./X_max)
dev_datagen.fit(X_dev)


# (note that we assume square images here)
width = X.shape[1]
height = width
channels = 3 # rgb data

nb_epoch = 100
batch_size = 128

# training callbacks
es = EarlyStopping(patience = 10)
ckpt = ModelCheckpoint("./trained/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only = True)
if not os.path.exists("./trained"):
    os.mkdir("./trained")

# build the network
images = Input(shape = (height, width, channels))
# we generate images, but we just need to flatten them since we only
# use fully connected layers
x = Flatten()(images)
x = Dense(64)(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)   # binary objective, so one classification neuron

model = Model(inputs = images, outputs = predictions)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_datagen.flow(X_train, y_train, batch_size = batch_size), steps_per_epoch = len(X_train)//batch_size, epochs = nb_epoch,
    validation_data = dev_datagen.flow(X_dev, y_dev, batch_size = batch_size), validation_steps = len(X_dev)//batch_size, callbacks = [es, ckpt])


# delete all but the last saved checkpoint, it has the best quality model      
files = [os.path.join("./trained", f) for f in os.listdir("./trained")] # add path to each file
files.sort(key = lambda x: os.path.getmtime(x))
for f in files[1:]:
    os.remove(f)




