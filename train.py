# Convolution neural network classifier for frozen/non frozen lakes
# Trained with Keras on TensorFlow backend (image dim ordering is (height, width, channels))

from __future__ import print_function

import joblib

from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D

summer_data = joblib.load("./data/converted_summer.joblib")
winter_data = joblib.load("./data/converted_winter.joblib")

# (note that we assume square images here)
width = summer_data.shape[1]
height = width
channels = 3 # rgb

nb_epoch = 100

# Build feature extraction layers
# This dataset is small, so we can't build a super powerful network,
# or we'll overfit.
images = Input(shape = (height, width, channels))
x = Convolution2D(8, 3, 3)(images)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Convolution2D(8, 3, 3)(images)
x = Activation('relu')(x)

# Build classification layers
x = Flatten()(x)
x = Dense(64)(x)
predictions = Dense(2, 'sigmoid')(x)   # one neuron per output class

model = Model(input = images, output = predictions)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

