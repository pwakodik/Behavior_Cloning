import os
import csv
import cv2
import math
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Dropout, Flatten
from keras.layers.convolutional import Cropping2D, Convolution2D
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from workspace_utils import active_session
from scipy import ndimage

from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D

### Defining Swish Activation ###
"""
from keras.backend import sigmoid
def swish(x, beta=1):
    return(x*sigmoid(beta*x))
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})
"""
"""
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return(x*sigmoid(x))

get_custom_objects().update({'swish':Swish(swish)})
"""
# Reading the dataset
imgdata = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)      # The first row in driving_log.csv contains title and not useful as steering value, so moving to next row data.
    for line in reader:
        imgdata.append(line)
        
# Splitting dataset into 80% Training data and 20% Validation data
train_samples, validation_samples = train_test_split(imgdata, test_size=0.2)

# Using Generator for training model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Shuffling the Dataset
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Iterating for Centre, Left and Right Camera images
                for i in range (0,3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    # Reading image in RGB
                    image = ndimage.imread(current_path)
                    images.append(image)

                    # Using correction factor to correct steering angles for Left and Right Camera images
                    steering_correction = 0.2
                    if i == 1:
                        measurement = float(batch_sample[3])+steering_correction
                    elif i == 2:
                        measurement = float(batch_sample[3])-steering_correction
                    else:
                        measurement = float(batch_sample[3])         

                    measurements.append(measurement)

            # Data Augmentation: Flipping the image around vertical axis to create more dataset
            aug_images, aug_measurements = [], []
            for image, measurement in zip(images, measurements):
                aug_images.append(image)
                aug_measurements.append(measurement)
                aug_images.append(cv2.flip(image,1))
                # correcting Steering angle measurement for flipped image
                aug_measurements.append(measurement*-1.0)

            X_train = np.array(aug_images)
            Y_train = np.array(aug_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)

# Set our batch size
batch_size=32

with active_session():
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    # Creating model similar to Nvidia Self driving Car model [Can be found at: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf]
    model = Sequential()
    # Normalizing Data
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    # Cropping top 75 pixels and bottom 25 pixels of image as they dont contain any useful data for driving
    model.add(Cropping2D(cropping=((70,25),(0,0))))    
    """
    ### Swish Activation ###
    model.add(Convolution2D(24, (5,5), strides=(2,2), activation="swish"))
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="swish"))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="swish"))
    # Adding Dropout Layer to reduce Overfitting
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3,3), activation="swish"))
    model.add(Convolution2D(64, (3,3), activation="swish"))
    """
    """
    ### Relu Activitation ###
    model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="relu"))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
    # Adding Dropout Layer to reduce Overfitting
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3,3), activation="relu"))
    model.add(Convolution2D(64, (3,3), activation="relu"))
    """
    """
    ### ELU Activation ###
    model.add(Convolution2D(24, (5,5), strides=(2,2), activation="elu"))
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="elu"))
    # Adding Dropout Layer to reduce Overfitting
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3,3), activation="elu"))
    model.add(Convolution2D(64, (3,3), activation="elu"))
    """
    """
    ### Reduced Layers ###
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="elu"))
    # Adding Dropout Layer to reduce Overfitting
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3,3), activation="elu"))
    model.add(Dropout(0.25))
    """
    ### With Reduced Conv2D layers, additional Dropout and MaxPooling 2D Layers ###
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="elu"))
    # Adding Dropout Layer to reduce Overfitting
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3,3), activation="elu"))
    model.add(Dropout(0.25))
    # Flattening Tensor before adding to Dense Layer
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    # Output Layer which is Steering value
    model.add(Dense(1))

    # Using Mean Square Error Loss function and Adam Optimiser for compiling Model
    model.compile(loss='mse', optimizer='adam')
    
    # Calling the Generator
    model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)
    
    # Saving the Model
    model.save('model.h5')
    # Getting Model Summary
    model.summary()