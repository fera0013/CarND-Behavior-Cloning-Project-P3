import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D
import data_utility
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def resize(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (66, 200))

def normalize(image):
    return image / 255.0 - 0.5

def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((22, 0), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(resize))
    model.add(Lambda(normalize))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))
    model.add(SpatialDropout2D(0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.001), loss="mse")

    return model

if __name__ == "__main__":
    batch_size = 128
    number_of_epochs = 5
    data = data_utility.get_steering_angle_data(0.025)
    train, val = train_test_split(data, test_size=0.2, random_state=0)
    train = train[0:19200]
    val = val[0:4800] 
    model = get_model()
    print(model.summary())
    model.fit_generator(
        data_utility.gen(train.image, train.steering, batch_size),
        samples_per_epoch = len(train),
        nb_epoch=number_of_epochs,
        verbose=1,
        validation_data = data_utility.gen(val.image, val.steering,batch_size),
        nb_val_samples = len(val)
    )

    print("Saving model, weights and configuration file.")
  
    model.save_weights('model.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)